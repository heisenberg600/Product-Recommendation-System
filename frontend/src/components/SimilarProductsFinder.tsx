import { useState } from 'react';
import { Search, X, Package, Star, Shield, Loader2 } from 'lucide-react';
import { usePopularItems } from '../hooks/useRecommendations';
import { apiClient } from '../api/client';

interface SimilarProductsFinderProps {
  onClose: () => void;
}

interface SimilarItem {
  item_id: string;
  relevance_score: number;
  confidence_score: number;
  item_price: number;
}

export function SimilarProductsFinder({ onClose }: SimilarProductsFinderProps) {
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [similarItems, setSimilarItems] = useState<SimilarItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { data: popularData } = usePopularItems(50);

  const items = popularData?.items || [];
  const filteredItems = items.filter((item) =>
    item.item_id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const findSimilar = async (itemId: string) => {
    setSelectedItem(itemId);
    setIsLoading(true);
    try {
      const response = await apiClient.getSimilarItems(itemId, 5);
      setSimilarItems(response.similar_items);
    } catch (error) {
      console.error('Error finding similar items:', error);
      setSimilarItems([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-3xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                <Search className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-white font-bold text-xl">Similar Products Finder</h2>
                <p className="text-white/70 text-sm">Select a product to find similar items</p>
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
            {/* Product Selection */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-3">
                Select a Product
              </label>

              {/* Search Input */}
              <div className="relative mb-4">
                <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                <input
                  type="text"
                  placeholder="Search products..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full py-2.5 pl-11 pr-4 rounded-xl border-2 border-slate-200 bg-white text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 transition-all"
                />
              </div>

              {/* Product List */}
              <div className="max-h-80 overflow-y-auto space-y-2">
                {filteredItems.map((item) => (
                  <button
                    key={item.item_id}
                    onClick={() => findSimilar(item.item_id)}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all ${
                      selectedItem === item.item_id
                        ? 'bg-emerald-50 ring-2 ring-emerald-500'
                        : 'bg-slate-50 hover:bg-slate-100'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                        selectedItem === item.item_id
                          ? 'bg-emerald-500 text-white'
                          : 'bg-slate-200 text-slate-500'
                      }`}>
                        <Package className="w-5 h-5" />
                      </div>
                      <div className="text-left">
                        <div className="font-mono text-sm font-medium text-slate-800">{item.item_id}</div>
                        <div className="text-xs text-slate-500">
                          ${item.item_price?.toFixed(2) || '0.00'}
                        </div>
                      </div>
                    </div>
                    {selectedItem === item.item_id && (
                      <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center">
                        <Search className="w-3 h-3 text-white" />
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Similar Items Results */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-3">
                Similar Products {selectedItem && `for ${selectedItem}`}
              </label>

              {isLoading ? (
                <div className="flex items-center justify-center py-16">
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="w-8 h-8 text-emerald-500 animate-spin" />
                    <span className="text-slate-500 text-sm">Finding similar items...</span>
                  </div>
                </div>
              ) : similarItems.length > 0 ? (
                <div className="space-y-3">
                  {similarItems.map((item, index) => (
                    <div
                      key={item.item_id}
                      className="p-4 rounded-xl bg-gradient-to-r from-slate-50 to-slate-100 border border-slate-200"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center text-white font-bold shadow-lg shadow-emerald-500/25">
                            {index + 1}
                          </div>
                          <div>
                            <div className="font-mono text-sm font-semibold text-slate-800">{item.item_id}</div>
                            <div className="text-xs text-slate-500">Similar to selected item</div>
                          </div>
                        </div>
                        {item.item_price > 0 && (
                          <div className="text-lg font-bold text-slate-800">${item.item_price.toFixed(2)}</div>
                        )}
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1 text-slate-500">
                              <Star className="w-3 h-3 text-amber-500" /> Similarity
                            </span>
                            <span className="font-semibold text-amber-600">{Math.round(item.relevance_score * 100)}%</span>
                          </div>
                          <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full"
                              style={{ width: `${item.relevance_score * 100}%` }}
                            />
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1 text-slate-500">
                              <Shield className="w-3 h-3 text-emerald-500" /> Confidence
                            </span>
                            <span className="font-semibold text-emerald-600">{Math.round(item.confidence_score * 100)}%</span>
                          </div>
                          <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full"
                              style={{ width: `${item.confidence_score * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : selectedItem ? (
                <div className="flex items-center justify-center py-16 text-slate-400">
                  <div className="text-center">
                    <Package className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">No similar items found</p>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center py-16 text-slate-400">
                  <div className="text-center">
                    <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Select a product to find similar items</p>
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
