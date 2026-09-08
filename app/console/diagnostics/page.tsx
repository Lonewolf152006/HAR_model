import { redirect } from "next/navigation";

export default function DiagnosticsRedirectPage() {
  redirect("/console/live-feed");
}
