import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Header } from './components/Header';
import { UserSelector } from './components/UserSelector';
import { ModelSelector } from './components/ModelSelector';
import { RecommendationResults } from './components/RecommendationResults';
import { PopularItems } from './components/PopularItems';
import { useRecommendations } from './hooks/useRecommendations';
import type { ModelType } from './types';
import { Sparkles } from 'lucide-react';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function RecommendationApp() {
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType | null>(null);

  const {
    data: recommendations,
    isLoading,
    error,
  } = useRecommendations(selectedUser, {
    n: 5,
    model: selectedModel || undefined,
    excludePurchased: true,
  });

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <div className="lg:col-span-4 space-y-5">
            <UserSelector
              selectedUser={selectedUser}
              onSelectUser={setSelectedUser}
            />
            <ModelSelector
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
            />
            <PopularItems />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-8">
            <RecommendationResults
              data={recommendations}
              isLoading={isLoading}
              error={error}
              selectedUser={selectedUser}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <Sparkles className="w-4 h-4" />
              <span>RecoAI - AI-Powered Product Recommendations</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span className="px-2 py-1 rounded bg-slate-100">FastAPI</span>
              <span className="px-2 py-1 rounded bg-slate-100">React</span>
              <span className="px-2 py-1 rounded bg-slate-100">5 ML Models</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RecommendationApp />
    </QueryClientProvider>
  );
}

export default App;
