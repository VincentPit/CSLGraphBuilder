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
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen antialiased">
        <Providers>
          <div className="flex min-h-screen">
            <Nav />
            <main className="flex-1 overflow-auto">
              <div className="px-8 py-8 lg:px-12 lg:py-12 max-w-[1280px] fade-up">
                {children}
              </div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
