import { Package, Star, Shield, Brain, TrendingUp, DollarSign, Layers, Zap } from 'lucide-react';
import type { RecommendationItem, ModelType } from '../types';

interface RecommendationCardProps {
  recommendation: RecommendationItem;
  rank: number;
}

const modelIcons: Record<ModelType, React.ReactNode> = {
  hybrid: <Layers className="w-3.5 h-3.5" />,
  item_cf: <Brain className="w-3.5 h-3.5" />,
  matrix_factorization: <Zap className="w-3.5 h-3.5" />,
  popularity: <TrendingUp className="w-3.5 h-3.5" />,
  price_segment: <DollarSign className="w-3.5 h-3.5" />,
};

const modelStyles: Record<ModelType, { bg: string; text: string; label: string }> = {
  hybrid: { bg: 'bg-blue-100', text: 'text-blue-700', label: 'Hybrid' },
  item_cf: { bg: 'bg-emerald-100', text: 'text-emerald-700', label: 'Item-CF' },
  matrix_factorization: { bg: 'bg-orange-100', text: 'text-orange-700', label: 'Matrix' },
  popularity: { bg: 'bg-pink-100', text: 'text-pink-700', label: 'Popular' },
  price_segment: { bg: 'bg-slate-200', text: 'text-slate-700', label: 'Price' },
};

const rankStyles: Record<number, string> = {
  1: 'from-amber-400 via-yellow-400 to-amber-500 shadow-amber-400/40',
  2: 'from-slate-300 via-slate-200 to-slate-400 shadow-slate-400/30',
  3: 'from-orange-400 via-amber-600 to-orange-500 shadow-orange-400/30',
  4: 'from-blue-400 via-blue-500 to-blue-600 shadow-blue-400/30',
  5: 'from-purple-400 via-purple-500 to-purple-600 shadow-purple-400/30',
};

export function RecommendationCard({ recommendation, rank }: RecommendationCardProps) {
  const relevancePercent = Math.round(recommendation.relevance_score * 100);
  const confidencePercent = Math.round(recommendation.confidence * 100);
  const model = modelStyles[recommendation.model_used];
  const rankGradient = rankStyles[rank] || 'from-slate-400 to-slate-500 shadow-slate-400/30';

  return (
    <div className="card card-hover p-0 overflow-visible relative">
      {/* Large Rank Badge */}
      <div className={`absolute -top-3 -left-3 z-10 w-12 h-12 rounded-2xl bg-gradient-to-br ${rankGradient} flex items-center justify-center shadow-lg`}>
        <span className="text-white font-bold text-xl drop-shadow-sm">#{rank}</span>
      </div>

      <div className="p-5 pt-6 pl-12">
        {/* Header Row */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-3">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 flex items-center justify-center flex-shrink-0">
              <Package className="w-7 h-7 text-slate-400" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 font-mono text-sm tracking-tight">
                {recommendation.item_id}
              </h3>
              <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium ${model.bg} ${model.text}`}>
                {modelIcons[recommendation.model_used]}
                {model.label}
              </span>
            </div>
          </div>

          {recommendation.item_price && (
            <div className="text-right">
              <div className="text-2xl font-bold text-slate-800">
                ${recommendation.item_price.toFixed(2)}
              </div>
            </div>
          )}
        </div>

        {/* Recommendation Reason */}
        <p className="text-sm text-slate-600 mb-4 leading-relaxed">
          {recommendation.recommendation_reason}
        </p>

        {/* Score Bars */}
        <div className="grid grid-cols-2 gap-4">
          {/* Relevance Score */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-medium text-slate-600">
                <Star className="w-3.5 h-3.5 text-amber-500" />
                Relevance
              </div>
              <span className="text-xs font-bold text-amber-600">{relevancePercent}%</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full transition-all duration-700 ease-out"
                style={{ width: `${relevancePercent}%` }}
              />
            </div>
          </div>

          {/* Confidence Score */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-medium text-slate-600">
                <Shield className="w-3.5 h-3.5 text-emerald-500" />
                Confidence
              </div>
              <span className="text-xs font-bold text-emerald-600">{confidencePercent}%</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full transition-all duration-700 ease-out"
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
