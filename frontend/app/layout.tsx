import type { Metadata } from "next";
import { JetBrains_Mono, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import AppShell from "@/components/layout/AppShell";
import { TickerProvider } from "@/lib/ticker-context";

const sans = Plus_Jakarta_Sans({ variable: "--font-sans-app", subsets: ["latin"] });
const mono = JetBrains_Mono({ variable: "--font-mono-app", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "MARS · PSX Stock Analysis Intelligence",
  description: "Multi-agent stock analysis for the Pakistan Stock Exchange",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full antialiased`}>
      <body className="h-full">
        <TickerProvider>
          <AppShell>{children}</AppShell>
        </TickerProvider>
      </body>
    </html>
  );
}
