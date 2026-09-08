// Pure Web Audio API avionics sound synthesizer
// Requires zero external sound assets, provides instantaneous avionics audio feedback

class AvionicsSoundSystem {
  private ctx: AudioContext | null = null;
  private isMuted: boolean = false;

  private getContext(): AudioContext | null {
    if (typeof window === "undefined") return null;
    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  private sirenInterval: NodeJS.Timeout | null = null;
  private currentSirenNodes: { osc: OscillatorNode; gain: GainNode }[] = [];

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    if (muted) {
      this.stopAlertSiren();
    }
  }

  public getMuted(): boolean {
    return this.isMuted;
  }

  /**
   * Accepted transition: crisp high-pitched avionics blip (1200Hz)
   */
  public playAcceptedTone() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(1200, now);
      osc.frequency.exponentialRampToValueAtTime(1350, now + 0.08);

      gain.gain.setValueAtTime(0.08, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.08);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.09);
    } catch {
      // Audio context might be restricted before user interaction
    }
  }

  /**
   * Rejected transition / gate failure: dull low-frequency rejection buzz (280Hz)
   */
  public playRejectedTone() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(280, now);
      osc.frequency.linearRampToValueAtTime(220, now + 0.12);

      gain.gain.setValueAtTime(0.09, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.13);
    } catch {
      // Audio context might be restricted
    }
  }

  /**
   * Cycle complete: dual-tone confirmation chime (800Hz -> 1400Hz)
   */
  public playCycleCompleteChime() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;

      // Note 1: 880Hz (A5)
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(880, now);
      gain1.gain.setValueAtTime(0.08, now);
      gain1.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.15);

      // Note 2: 1320Hz (E6)
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(1320, now + 0.12);
      gain2.gain.setValueAtTime(0.09, now + 0.12);
      gain2.gain.exponentialRampToValueAtTime(0.0001, now + 0.32);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.12);
      osc2.stop(now + 0.34);
    } catch {
      // Audio context might be restricted
    }
  }

  /**
   * Start sustained two-tone console alarm siren:
   * Repeating 2-3 alternating tone pulses (920Hz -> 700Hz -> 920Hz)
   * Repeats every 1000ms until stopAlertSiren() is called or system is muted.
   * If already running, does NOT stack/layer multiple sirens.
   */
  public startAlertSiren() {
    if (this.isMuted) return;
    // Prevent overlapping/stacking multiple sirens
    if (this.sirenInterval) return;

    const playSirenBurst = () => {
      if (this.isMuted) {
        this.stopAlertSiren();
        return;
      }
      const ctx = this.getContext();
      if (!ctx) return;

      try {
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        // Authentic industrial avionics two-tone square wave
        osc.type = "square";
        osc.frequency.setValueAtTime(920, now);
        osc.frequency.setValueAtTime(700, now + 0.12);
        osc.frequency.setValueAtTime(920, now + 0.24);

        // Crisp envelope for the 3 alternating pulses (total ~360ms tone, 640ms rest)
        gain.gain.setValueAtTime(0.07, now);
        gain.gain.setValueAtTime(0.07, now + 0.36);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.40);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now);
        osc.stop(now + 0.41);

        this.currentSirenNodes.push({ osc, gain });
        setTimeout(() => {
          this.currentSirenNodes = this.currentSirenNodes.filter((n) => n.osc !== osc);
        }, 450);
      } catch {
        // AudioContext might be restricted before interaction
      }
    };

    // Play first burst immediately
    playSirenBurst();
    // Repeat every 1.0s in sync with visual alert pulse
    this.sirenInterval = setInterval(playSirenBurst, 1000);
  }

  /**
   * Immediately stops the repeating alert siren and cuts any ringing tone.
   */
  public stopAlertSiren() {
    if (this.sirenInterval) {
      clearInterval(this.sirenInterval);
      this.sirenInterval = null;
    }
    for (const node of this.currentSirenNodes) {
      try {
        node.gain.gain.setValueAtTime(0, 0);
        node.osc.stop();
        node.osc.disconnect();
      } catch {}
    }
    this.currentSirenNodes = [];
  }

  /**
   * Play alert: triggers repeating siren
   */
  public playAlertTone() {
    this.startAlertSiren();
  }

  /**
   * Spoken alert via Web Speech API (if supported and unmuted)
   */
  public speakVoiceAlert(text: string) {
    if (this.isMuted || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.1;
      utterance.pitch = 0.95;
      utterance.volume = 0.8;
      window.speechSynthesis.speak(utterance);
    } catch {
      // Speech synthesis fallback
    }
  }
}

export const avionicsAudio = new AvionicsSoundSystem();
