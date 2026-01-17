import { useState } from 'react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { Header } from './components/Header';
import { UserSelector } from './components/UserSelector';
import { ModelSelector } from './components/ModelSelector';
import { RecommendationResults } from './components/RecommendationResults';
import { PopularItems } from './components/PopularItems';
import { SimilarItems } from './components/SimilarItems';
import { CustomScenario } from './components/CustomScenario';
import { useRecommendations } from './hooks/useRecommendations';
import { apiClient } from './api/client';
import type { ModelType } from './types';
import { Sparkles, Wand2, UserX, Package, Star, Shield, Search } from 'lucide-react';
import { SimilarProductsFinder } from './components/SimilarProductsFinder';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Anonymous recommendations component
function AnonymousResults({ onShowSimilar }: { onShowSimilar: (itemId: string) => void }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['anonymous-recommendations'],
    queryFn: () => apiClient.getAnonymousRecommendations(5),
  });

  if (isLoading) {
    return (
      <div className="card p-16 text-center border-0 shadow-xl">
        <div className="w-20 h-20 mx-auto mb-6 rounded-3xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center animate-pulse">
          <Sparkles className="w-10 h-10 text-white" />
        </div>
        <h3 className="text-xl font-semibold text-slate-800 mb-2">Loading recommendations...</h3>
        <p className="text-slate-500">Finding popular items for new visitors</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="card p-16 text-center border-0 shadow-xl bg-red-50">
        <h3 className="text-xl font-semibold text-red-800 mb-2">Error loading recommendations</h3>
        <p className="text-red-600">Please try again later</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Anonymous User Banner */}
      <div className="card p-5 bg-gradient-to-r from-violet-500 via-purple-500 to-pink-500 border-0 shadow-xl shadow-purple-500/20">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
            <UserX className="w-7 h-7 text-white" />
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">New Visitor</h3>
            <p className="text-white/80 text-sm">Showing popular items recommended for first-time visitors</p>
          </div>
        </div>
      </div>

      {/* Results Header */}
      <div className="flex items-center gap-3 px-1">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/25">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-800">Top {data.recommendations.length} Popular Items</h2>
          <p className="text-sm text-slate-500">Based on trending products</p>
        </div>
      </div>

      {/* Recommendations Grid */}
      <div className="grid gap-5">
        {data.recommendations.map((rec, index) => (
          <div key={rec.item_id} className="card card-hover p-0 overflow-visible relative border-0 shadow-lg">
            {/* Rank Badge */}
            <div className={`absolute -top-3 -left-3 z-10 w-12 h-12 rounded-2xl bg-gradient-to-br ${
              index === 0 ? 'from-amber-400 to-yellow-500' :
              index === 1 ? 'from-slate-300 to-slate-400' :
              index === 2 ? 'from-orange-400 to-amber-500' :
              'from-violet-400 to-purple-500'
            } flex items-center justify-center shadow-lg`}>
              <span className="text-white font-bold text-xl">#{index + 1}</span>
            </div>

            <div className="p-5 pt-6 pl-14">
              <div className="flex items-start justify-between gap-3 mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 flex items-center justify-center">
                    <Package className="w-7 h-7 text-slate-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-slate-900 font-mono text-sm">{rec.item_id}</h3>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-pink-100 text-pink-700">
                      Popular
                    </span>
                  </div>
                </div>
                {rec.item_price > 0 && (
                  <div className="text-2xl font-bold text-slate-800">${rec.item_price.toFixed(2)}</div>
                )}
              </div>

              <p className="text-sm text-slate-600 mb-4">{rec.recommendation_reason}</p>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 text-xs font-medium text-slate-600">
                      <Star className="w-3.5 h-3.5 text-amber-500" /> Relevance
                    </div>
                    <span className="text-xs font-bold text-amber-600">{Math.round(rec.relevance_score * 100)}%</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full" style={{ width: `${rec.relevance_score * 100}%` }} />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 text-xs font-medium text-slate-600">
                      <Shield className="w-3.5 h-3.5 text-emerald-500" /> Confidence
                    </div>
                    <span className="text-xs font-bold text-emerald-600">{Math.round(rec.confidence_score * 100)}%</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full" style={{ width: `${rec.confidence_score * 100}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RecommendationApp() {
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType | null>(null);
  const [similarItemId, setSimilarItemId] = useState<string | null>(null);
  const [isAnonymous, setIsAnonymous] = useState(false);
  const [showCustomScenario, setShowCustomScenario] = useState(false);
  const [showSimilarFinder, setShowSimilarFinder] = useState(false);
  const [isLoyalUser, setIsLoyalUser] = useState(false);

  const handleShowSimilar = (itemId: string) => {
    setSimilarItemId(itemId);
  };

  const handleCloseSimilar = () => {
    setSimilarItemId(null);
  };

  const handleSelectAnonymous = () => {
    setIsAnonymous(true);
    setSelectedUser(null);
    setIsLoyalUser(false);
  };

  const handleSelectUser = (userId: string | null, loyal: boolean = false) => {
    setSelectedUser(userId);
    setIsLoyalUser(loyal);
    if (userId) {
      setIsAnonymous(false);
    }
  };

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
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 via-white to-slate-100">
      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <div className="lg:col-span-4 space-y-5">
            <UserSelector
              selectedUser={selectedUser}
              onSelectUser={handleSelectUser}
              onSelectAnonymous={handleSelectAnonymous}
              isAnonymous={isAnonymous}
            />

            {/* Custom Scenario Button */}
            <button
              onClick={() => setShowCustomScenario(true)}
              className="w-full card p-4 border-0 shadow-xl shadow-indigo-500/10 bg-gradient-to-r from-indigo-50 via-purple-50 to-pink-50 hover:from-indigo-100 hover:via-purple-100 hover:to-pink-100 transition-all group"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/25 group-hover:scale-110 transition-transform">
                  <Wand2 className="w-5 h-5 text-white" />
                </div>
                <div className="text-left">
                  <div className="font-semibold text-slate-800">Custom Scenario</div>
                  <div className="text-xs text-slate-500">Build your own purchase history</div>
                </div>
              </div>
            </button>

            {/* Similar Products Finder Button */}
            <button
              onClick={() => setShowSimilarFinder(true)}
              className="w-full card p-4 border-0 shadow-xl shadow-slate-200/50 bg-gradient-to-r from-emerald-50 via-teal-50 to-cyan-50 hover:from-emerald-100 hover:via-teal-100 hover:to-cyan-100 transition-all group"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 via-teal-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-110 transition-transform">
                  <Search className="w-5 h-5 text-white" />
                </div>
                <div className="text-left">
                  <div className="font-semibold text-slate-800">Similar Products</div>
                  <div className="text-xs text-slate-500">Find products like any item</div>
                </div>
              </div>
            </button>

            {/* Only show ModelSelector for loyal users */}
            {selectedUser && isLoyalUser && (
              <ModelSelector
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
              />
            )}
            <PopularItems onShowSimilar={handleShowSimilar} />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-8">
            {isAnonymous ? (
              <AnonymousResults onShowSimilar={handleShowSimilar} />
            ) : (
              <RecommendationResults
                data={recommendations}
                isLoading={isLoading}
                error={error}
                selectedUser={selectedUser}
                onShowSimilar={handleShowSimilar}
              />
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200/50 bg-white/80 backdrop-blur mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                <Sparkles className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="font-medium">RecoAI</span>
              <span className="text-slate-400">- AI-Powered Product Recommendations</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-lg bg-gradient-to-r from-green-50 to-emerald-50 text-emerald-700 font-medium border border-emerald-200">FastAPI</span>
              <span className="px-2.5 py-1 rounded-lg bg-gradient-to-r from-blue-50 to-cyan-50 text-blue-700 font-medium border border-blue-200">React</span>
              <span className="px-2.5 py-1 rounded-lg bg-gradient-to-r from-purple-50 to-pink-50 text-purple-700 font-medium border border-purple-200">ML Models</span>
            </div>
          </div>
        </div>
      </footer>

      {/* Similar Items Modal */}
      {similarItemId && (
        <SimilarItems itemId={similarItemId} onClose={handleCloseSimilar} />
      )}

      {/* Custom Scenario Modal */}
      {showCustomScenario && (
        <CustomScenario onClose={() => setShowCustomScenario(false)} />
      )}

      {/* Similar Products Finder Modal */}
      {showSimilarFinder && (
        <SimilarProductsFinder onClose={() => setShowSimilarFinder(false)} />
      )}
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
