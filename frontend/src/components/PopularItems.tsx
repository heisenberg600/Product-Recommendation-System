import { TrendingUp, Package, Users, ShoppingCart, Search } from 'lucide-react';
import { usePopularItems } from '../hooks/useRecommendations';

interface PopularItemsProps {
  onShowSimilar?: (itemId: string) => void;
}

const rankColors = [
  'from-amber-400 to-yellow-500 text-white',
  'from-slate-300 to-slate-400 text-slate-700',
  'from-orange-400 to-amber-500 text-white',
  'from-blue-400 to-blue-500 text-white',
  'from-purple-400 to-purple-500 text-white',
];

export function PopularItems({ onShowSimilar }: PopularItemsProps) {
  const { data, isLoading } = usePopularItems(10);

  return (
    <div className="card p-5">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-pink-600 flex items-center justify-center shadow-lg shadow-rose-500/25">
          <TrendingUp className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-800">Trending</h2>
          <p className="text-xs text-slate-500">Top selling products</p>
        </div>
      </div>

      <div className="space-y-2">
        {isLoading ? (
          Array(5)
            .fill(0)
            .map((_, i) => (
              <div key={i} className="skeleton h-14 rounded-lg" />
            ))
        ) : (
          data?.items.slice(0, 5).map((item, index) => (
            <div
              key={item.item_id}
              className="flex items-center gap-3 p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 transition-all group"
            >
              <div className={`w-7 h-7 rounded-lg bg-gradient-to-br ${rankColors[index]} flex items-center justify-center font-bold text-xs shadow-sm`}>
                {index + 1}
              </div>
              <div className="w-9 h-9 rounded-lg bg-white border border-slate-200 flex items-center justify-center flex-shrink-0">
                <Package className="w-4 h-4 text-slate-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-mono text-xs text-slate-800 truncate">
                  {item.item_id}
                </div>
                <div className="text-xs font-medium text-slate-500">
                  ${item.item_price?.toFixed(2) || 'N/A'}
                </div>
              </div>
              <div className="text-right flex-shrink-0 flex items-center gap-2">
                <div>
                  <div className="flex items-center gap-1 text-xs font-medium text-slate-600">
                    <ShoppingCart className="w-3 h-3" />
                    {item.purchase_count}
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-slate-400">
                    <Users className="w-2.5 h-2.5" />
                    {item.unique_buyers} buyers
                  </div>
                </div>
                {onShowSimilar && (
                  <button
                    onClick={() => onShowSimilar(item.item_id)}
                    className="w-7 h-7 rounded-lg bg-indigo-100 hover:bg-indigo-200 flex items-center justify-center transition-colors opacity-0 group-hover:opacity-100"
                    title="Find similar items"
                  >
                    <Search className="w-3.5 h-3.5 text-indigo-600" />
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
