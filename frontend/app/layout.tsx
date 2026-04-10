import type { Metadata } from 'next';
import './globals.css';
import Nav from '@/components/Nav';
import Providers from '@/components/Providers';

export const metadata: Metadata = {
  title: 'GraphBuilder',
  description: 'Knowledge-graph construction and curation interface',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0f1e] text-slate-200 antialiased">
        <Providers>
          <div className="flex min-h-screen">
            <Nav />
            <main className="flex-1 p-8 lg:p-12 xl:p-16 overflow-auto">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
