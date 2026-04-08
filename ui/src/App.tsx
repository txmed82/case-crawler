import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import IngestPage from './pages/IngestPage';
import SearchPage from './pages/SearchPage';
import SourcesPage from './pages/SourcesPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
    isActive
      ? 'bg-blue-100 text-blue-700'
      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
  }`;

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-50">
          <nav className="border-b border-gray-200 bg-white shadow-sm">
            <div className="mx-auto max-w-4xl px-4 sm:px-6">
              <div className="flex h-14 items-center justify-between">
                <div className="flex items-center gap-1">
                  <span className="mr-4 text-base font-bold text-gray-900 tracking-tight">
                    CaseCrawler
                  </span>
                  <NavLink to="/" end className={navLinkClass}>
                    Ingest
                  </NavLink>
                  <NavLink to="/search" className={navLinkClass}>
                    Search
                  </NavLink>
                  <NavLink to="/sources" className={navLinkClass}>
                    Sources
                  </NavLink>
                </div>
              </div>
            </div>
          </nav>

          <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
            <Routes>
              <Route path="/" element={<IngestPage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/sources" element={<SourcesPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
