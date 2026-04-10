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
      <body className="min-h-screen text-slate-900 antialiased">
        <Providers>
          <div className="flex min-h-screen">
            <Nav />
            <main className="flex-1 p-6 lg:p-8 overflow-auto">
              <div className="max-w-[1200px] mx-auto">{children}</div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
