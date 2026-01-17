import { useState } from 'react';
import { ShoppingCart, X, Sparkles, Calendar, Loader2, Package, Star, Shield, Search, Check } from 'lucide-react';
import { apiClient } from '../api/client';
import { usePopularItems } from '../hooks/useRecommendations';

interface CustomScenarioProps {
  onClose: () => void;
}

interface CustomRecommendation {
  item_id: string;
  relevance_score: number;
  confidence_score: number;
  item_price: number;
  recommendation_reason: string;
  model_source: string;
}

export function CustomScenario({ onClose }: CustomScenarioProps) {
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [timestamp, setTimestamp] = useState('');
  const [recommendations, setRecommendations] = useState<CustomRecommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const { data: popularData } = usePopularItems(100);

  const availableItems = popularData?.items || [];
  const filteredItems = availableItems.filter(
    (item) =>
      item.item_id.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !selectedItems.includes(item.item_id)
  );

  const addItem = (itemId: string) => {
    if (!selectedItems.includes(itemId)) {
      setSelectedItems([...selectedItems, itemId]);
      setSearchQuery('');
    }
  };

  const removeItem = (itemToRemove: string) => {
    setSelectedItems(selectedItems.filter(item => item !== itemToRemove));
  };

  const getRecommendations = async () => {
    setIsLoading(true);
    try {
      const response = await apiClient.getCustomRecommendations(
        selectedItems,
        5,
        timestamp || undefined
      );
      setRecommendations(response.recommendations);
      setMessage(response.message);
    } catch (error) {
      console.error('Error getting recommendations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-3xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-white font-bold text-xl">Custom Scenario</h2>
                <p className="text-white/70 text-sm">Build your own purchase history</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="w-10 h-10 rounded-xl bg-white/20 hover:bg-white/30 flex items-center justify-center transition-colors"
            >
              <X className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-180px)]">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Product Selection */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-3">
                Select Products for Cart
              </label>

              {/* Search Input */}
              <div className="relative mb-4">
                <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                <input
                  type="text"
                  placeholder="Search products to add..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full py-2.5 pl-11 pr-4 rounded-xl border-2 border-slate-200 bg-white text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 transition-all"
                />
              </div>

              {/* Available Products List */}
              <div className="max-h-48 overflow-y-auto space-y-1.5 mb-4">
                {filteredItems.slice(0, 20).map((item) => (
                  <button
                    key={item.item_id}
                    onClick={() => addItem(item.item_id)}
                    className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-slate-50 hover:bg-indigo-50 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Package className="w-4 h-4 text-slate-400" />
                      <span className="font-mono text-sm text-slate-700">{item.item_id}</span>
                    </div>
                    <span className="text-xs text-slate-500">${item.item_price?.toFixed(2) || '0.00'}</span>
                  </button>
                ))}
                {filteredItems.length === 0 && searchQuery && (
                  <div className="text-center py-4 text-slate-400 text-sm">
                    No products found matching "{searchQuery}"
                  </div>
                )}
              </div>

              {/* Selected Items */}
              {selectedItems.length > 0 && (
                <div className="mb-4">
                  <label className="block text-sm font-semibold text-slate-700 mb-2">
                    <ShoppingCart className="w-4 h-4 inline mr-1" />
                    Selected Items ({selectedItems.length})
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {selectedItems.map((itemId) => (
                      <div
                        key={itemId}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200"
                      >
                        <Package className="w-4 h-4 text-indigo-500" />
                        <span className="font-mono text-sm text-slate-700">{itemId}</span>
                        <button
                          onClick={() => removeItem(itemId)}
                          className="w-5 h-5 rounded-full bg-red-100 hover:bg-red-200 flex items-center justify-center transition-colors"
                        >
                          <X className="w-3 h-3 text-red-600" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Timestamp Input */}
              <div className="mb-4">
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  <Calendar className="w-4 h-4 inline mr-1" />
                  Timestamp (Optional)
                </label>
                <input
                  type="datetime-local"
                  value={timestamp}
                  onChange={(e) => setTimestamp(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl border-2 border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 outline-none transition-all text-sm"
                />
              </div>

              {/* Get Recommendations Button */}
              <button
                onClick={getRecommendations}
                disabled={isLoading}
                className="w-full py-4 rounded-xl bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white font-semibold hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 transition-all shadow-lg shadow-purple-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Get Recommendations
                  </>
                )}
              </button>
            </div>

            {/* Right: Recommendations */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="block text-sm font-semibold text-slate-700">
                  Recommendations
                </label>
                {message && (
                  <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded-lg">{message}</span>
                )}
              </div>

              {recommendations.length > 0 ? (
                <div className="space-y-3">
                  {recommendations.map((rec, index) => (
                    <div
                      key={rec.item_id}
                      className="p-4 rounded-xl bg-gradient-to-r from-slate-50 to-slate-100 border border-slate-200 hover:shadow-md transition-all"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold shadow-lg shadow-indigo-500/25">
                            {index + 1}
                          </div>
                          <div>
                            <div className="font-mono text-sm font-semibold text-slate-800">{rec.item_id}</div>
                            <div className="text-xs text-slate-500">{rec.recommendation_reason}</div>
                          </div>
                        </div>
                        {rec.item_price > 0 && (
                          <div className="text-lg font-bold text-slate-800">${rec.item_price.toFixed(2)}</div>
                        )}
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1 text-slate-500">
                              <Star className="w-3 h-3 text-amber-500" /> Relevance
                            </span>
                            <span className="font-semibold text-amber-600">{Math.round(rec.relevance_score * 100)}%</span>
                          </div>
                          <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full"
                              style={{ width: `${rec.relevance_score * 100}%` }}
                            />
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1 text-slate-500">
                              <Shield className="w-3 h-3 text-emerald-500" /> Confidence
                            </span>
                            <span className="font-semibold text-emerald-600">{Math.round(rec.confidence_score * 100)}%</span>
                          </div>
                          <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full"
                              style={{ width: `${rec.confidence_score * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center py-16 text-slate-400">
                  <div className="text-center">
                    <Sparkles className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Select products and click<br/>"Get Recommendations"</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
