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
          href="https://fonts.googleapis.com/css2?family=Nunito:wght@500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen antialiased">
        <Providers>
          <div className="flex flex-col lg:flex-row min-h-screen">
            <Nav />
            <main className="flex-1 overflow-auto">
              <div className="px-10 py-10 sm:px-16 sm:py-14 lg:px-24 lg:py-16 max-w-[1320px] mx-auto fade-up">
                {children}
              </div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
