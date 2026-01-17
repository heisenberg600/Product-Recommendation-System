import { Clock, Zap, Crown, UserPlus, AlertCircle, Loader2, ShoppingBag, Package, Sparkles } from 'lucide-react';
import type { RecommendationResponse } from '../types';
import { RecommendationCard } from './RecommendationCard';

interface RecommendationResultsProps {
  data: RecommendationResponse | undefined;
  isLoading: boolean;
  error: Error | null;
  selectedUser: string | null;
  onShowSimilar?: (itemId: string) => void;
}

export function RecommendationResults({
  data,
  isLoading,
  error,
  selectedUser,
  onShowSimilar,
}: RecommendationResultsProps) {
  if (!selectedUser) {
    return (
      <div className="card p-16 text-center">
        <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center mx-auto mb-6 shadow-inner">
          <ShoppingBag className="w-10 h-10 text-slate-400" />
        </div>
        <h3 className="text-xl font-semibold text-slate-800 mb-3">
          Select a Customer
        </h3>
        <p className="text-slate-500 max-w-sm mx-auto leading-relaxed">
          Choose a user from the sidebar to generate personalized product recommendations
        </p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="card p-16 text-center">
        <div className="relative w-20 h-20 mx-auto mb-6">
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-blue-500 to-indigo-600 animate-pulse" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="w-10 h-10 text-white animate-spin" />
          </div>
        </div>
        <h3 className="text-xl font-semibold text-slate-800 mb-3">
          Generating Recommendations
        </h3>
        <p className="text-slate-500">
          Analyzing preferences and finding the best products...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-16 text-center bg-red-50 border-red-200">
        <div className="w-20 h-20 rounded-3xl bg-red-100 flex items-center justify-center mx-auto mb-6">
          <AlertCircle className="w-10 h-10 text-red-500" />
        </div>
        <h3 className="text-xl font-semibold text-red-900 mb-3">
          Error Loading Recommendations
        </h3>
        <p className="text-red-600 max-w-sm mx-auto">
          {error.message || 'Something went wrong. Please try again.'}
        </p>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const userInfo = data.user_info;

  return (
    <div className="space-y-6">
      {/* User Info Card */}
      <div className="card p-5 bg-gradient-to-r from-white to-slate-50">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg ${
                userInfo.user_type === 'loyal'
                  ? 'bg-gradient-to-br from-amber-400 to-orange-500 shadow-amber-500/30'
                  : 'bg-gradient-to-br from-emerald-400 to-teal-500 shadow-emerald-500/30'
              }`}
            >
              {userInfo.user_type === 'loyal' ? (
                <Crown className="w-7 h-7 text-white" />
              ) : (
                <UserPlus className="w-7 h-7 text-white" />
              )}
            </div>
            <div>
              <h3 className="font-bold text-slate-900 font-mono text-lg">
                {userInfo.user_id}
              </h3>
              <p className={`text-sm font-medium ${userInfo.user_type === 'loyal' ? 'text-amber-600' : 'text-emerald-600'}`}>
                {userInfo.user_type === 'loyal' ? 'Loyal Customer' : 'New Customer'}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-6">
            <div className="text-center px-4 py-2 rounded-xl bg-white shadow-sm border border-slate-100">
              <div className="font-bold text-slate-900 text-xl">
                {userInfo.total_purchases}
              </div>
              <div className="text-xs text-slate-500 font-medium">Purchases</div>
            </div>
            <div className="text-center px-4 py-2 rounded-xl bg-white shadow-sm border border-slate-100">
              <div className="font-bold text-slate-900 text-xl">
                {userInfo.unique_items}
              </div>
              <div className="text-xs text-slate-500 font-medium">Unique Items</div>
            </div>
            {userInfo.avg_item_price && (
              <div className="text-center px-4 py-2 rounded-xl bg-white shadow-sm border border-slate-100">
                <div className="font-bold text-slate-900 text-xl">
                  ${userInfo.avg_item_price.toFixed(0)}
                </div>
                <div className="text-xs text-slate-500 font-medium">Avg Price</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Header */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-800">
              Top {data.recommendations.length} Recommendations
            </h2>
            <p className="text-sm text-slate-500">
              Using <span className="font-semibold text-indigo-600">{data.primary_model.replace(/_/g, ' ')}</span>
              {data.fallback_used && (
                <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-md bg-amber-100 text-amber-700 text-xs font-medium">
                  Fallback
                </span>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-100 text-sm">
          <Clock className="w-4 h-4 text-slate-500" />
          <span className="font-medium text-slate-700">{data.processing_time_ms.toFixed(0)}ms</span>
          <Zap className="w-4 h-4 text-amber-500" />
        </div>
      </div>

      {/* Recommendations Grid */}
      <div className="grid gap-6 md:grid-cols-1 lg:grid-cols-1 xl:grid-cols-2">
        {data.recommendations.map((rec, index) => (
          <RecommendationCard
            key={rec.item_id}
            recommendation={rec}
            rank={index + 1}
            onShowSimilar={onShowSimilar}
          />
        ))}
      </div>

      {data.recommendations.length === 0 && (
        <div className="card p-16 text-center">
          <div className="w-20 h-20 rounded-3xl bg-slate-100 flex items-center justify-center mx-auto mb-6">
            <Package className="w-10 h-10 text-slate-400" />
          </div>
          <h3 className="text-xl font-semibold text-slate-800 mb-3">
            No Recommendations
          </h3>
          <p className="text-slate-500">
            No recommendations could be generated for this user.
          </p>
        </div>
      )}
    </div>
  );
}
