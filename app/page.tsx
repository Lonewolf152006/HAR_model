import Link from "next/link";
import { ShieldCheck, ArrowRight, Terminal } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0B0D10] technical-blueprint-bg text-[#E6E9ED] flex flex-col items-center justify-center p-6 relative font-mono select-none">
      <div className="film-grain" />
      <div className="scanline-overlay absolute inset-0 pointer-events-none opacity-20" />

      <div className="max-w-xl w-full bg-[#12151A] bezel-depth border border-white/10 p-8 rounded-[2px] shadow-2xl relative z-10 space-y-6">
        <div className="flex items-center justify-between border-b border-white/10 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-[2px] bg-[#171B21] border border-[#00E08A]/50 flex items-center justify-center">
              <ShieldCheck className="w-4 h-4 text-[#00E08A] glow-nominal" />
            </div>
            <div>
              <div className="text-xs font-bold tracking-wider text-[#E6E9ED]">
                ASTROFLOW AI
              </div>
              <div className="text-[10px] text-[#565C66]">
                BIO-ASTRONAUTICS RESEARCH DIVISION
              </div>
            </div>
          </div>
          <span className="px-2 py-0.5 text-[9px] font-semibold bg-[#171B21] text-[#00E08A] border border-[#00E08A]/40 glow-nominal rounded-[2px]">
            ONLINE: NOMINAL
          </span>
        </div>

        <div className="space-y-3 font-sans">
          <h1 className="text-xl font-bold tracking-tight text-[#E6E9ED]">
            Astronaut Human Activity Recognition Telemetry Console
          </h1>
          <p className="text-xs text-[#8A919C] leading-relaxed">
            Real-time on-board avionics dashboard monitoring the 5-gate DecisionStabilizer
            pipeline, geometric containment engine, and 7-step physical microgravity experiment protocol.
          </p>
        </div>

        <div className="p-3 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px] space-y-1.5 text-xs font-mono text-[#8A919C]">
          <div className="flex justify-between">
            <span>FLIGHT FIRMWARE:</span>
            <span className="text-[#E6E9ED]">v4.2.1-PROD</span>
          </div>
          <div className="flex justify-between">
            <span>HARDWARE TARGET:</span>
            <span className="text-[#E6E9ED]">ISS-COLUMBUS: NODE-ALPHA</span>
          </div>
          <div className="flex justify-between">
            <span>AI ENGINE:</span>
            <span className="text-[#00E08A] font-semibold">MediaPipe + YOLOv8 + BiLSTM-Attn</span>
          </div>
        </div>

        <div className="pt-2">
          <Link
            href="/console/live-feed"
            className="btn-bracket w-full py-3 bg-[#171B21] hover:bg-[#1E232B] border border-[#00E08A]/60 hover:border-[#00E08A] text-[#00E08A] text-sm font-bold flex items-center justify-center gap-2 rounded-[2px] transition-all tracking-wider bezel-depth-subtle"
          >
            <span>LAUNCH AVIONICS CONSOLE</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-white/5 text-[10px] text-[#565C66]">
          <span>BUILD PRIORITY: STEPS 1–3 ARMED</span>
          <span className="flex items-center gap-1">
            <Terminal className="w-3 h-3 text-[#565C66]" />
            <span>SOCKET PORT: 8554</span>
          </span>
        </div>
      </div>
    </main>
  );
}
