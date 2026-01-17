import { Package, Star, Shield, X, Loader2 } from 'lucide-react';
import { useSimilarItems } from '../hooks/useRecommendations';

interface SimilarItemsProps {
  itemId: string;
  onClose: () => void;
}

export function SimilarItems({ itemId, onClose }: SimilarItemsProps) {
  const { data, isLoading, error } = useSimilarItems(itemId, 5);

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-white font-semibold text-lg">Similar Items</h2>
            <p className="text-indigo-200 text-sm font-mono">{itemId}</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-colors"
          >
            <X className="w-5 h-5 text-white" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <p className="text-red-500">Failed to load similar items</p>
              <p className="text-slate-400 text-sm mt-1">Please try again later</p>
            </div>
          ) : data?.similar_items.length === 0 ? (
            <div className="text-center py-12">
              <Package className="w-12 h-12 text-slate-300 mx-auto mb-3" />
              <p className="text-slate-500">No similar items found</p>
            </div>
          ) : (
            <div className="space-y-4">
              {data?.similar_items.map((item, index) => (
                <div
                  key={item.item_id}
                  className="bg-slate-50 rounded-xl p-4 hover:bg-slate-100 transition-colors"
                >
                  <div className="flex items-start gap-4">
                    {/* Rank Badge */}
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center flex-shrink-0">
                      <span className="text-white font-bold text-sm">{index + 1}</span>
                    </div>

                    {/* Item Icon */}
                    <div className="w-12 h-12 rounded-xl bg-white border border-slate-200 flex items-center justify-center flex-shrink-0">
                      <Package className="w-6 h-6 text-slate-400" />
                    </div>

                    {/* Item Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-mono text-sm font-semibold text-slate-800 truncate">
                          {item.item_id}
                        </h3>
                        {item.item_price && (
                          <span className="text-lg font-bold text-slate-700">
                            ${item.item_price.toFixed(2)}
                          </span>
                        )}
                      </div>

                      {/* Scores */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-1 text-slate-500">
                              <Star className="w-3 h-3 text-amber-500" />
                              Relevance
                            </div>
                            <span className="font-semibold text-amber-600">
                              {Math.round(item.relevance_score * 100)}%
                            </span>
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
                            <div className="flex items-center gap-1 text-slate-500">
                              <Shield className="w-3 h-3 text-emerald-500" />
                              Confidence
                            </div>
                            <span className="font-semibold text-emerald-600">
                              {Math.round(item.confidence_score * 100)}%
                            </span>
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
                  </div>
                </div>
              ))}

              {/* Processing time */}
              {data && (
                <div className="text-center text-xs text-slate-400 pt-2">
                  Found in {data.processing_time_ms.toFixed(1)}ms
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
