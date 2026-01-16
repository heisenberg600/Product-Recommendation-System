import { Brain, Layers, TrendingUp, DollarSign, Sparkles, Zap, Check } from 'lucide-react';
import type { ModelType } from '../types';

interface ModelSelectorProps {
  selectedModel: ModelType | null;
  onSelectModel: (model: ModelType | null) => void;
}

const models: { key: ModelType | 'auto'; icon: React.ReactNode; name: string; description: string; gradient: string; bgColor: string }[] = [
  {
    key: 'auto',
    icon: <Sparkles className="w-5 h-5" />,
    name: 'Auto Select',
    description: 'System picks the best',
    gradient: 'from-violet-500 to-purple-600',
    bgColor: 'bg-violet-50',
  },
  {
    key: 'hybrid',
    icon: <Layers className="w-5 h-5" />,
    name: 'Hybrid',
    description: 'Combined approach',
    gradient: 'from-blue-500 to-cyan-500',
    bgColor: 'bg-blue-50',
  },
  {
    key: 'item_cf',
    icon: <Brain className="w-5 h-5" />,
    name: 'Item-CF',
    description: 'Similar items',
    gradient: 'from-emerald-500 to-teal-500',
    bgColor: 'bg-emerald-50',
  },
  {
    key: 'matrix_factorization',
    icon: <Zap className="w-5 h-5" />,
    name: 'Matrix Factor',
    description: 'Pattern learning',
    gradient: 'from-orange-500 to-amber-500',
    bgColor: 'bg-orange-50',
  },
  {
    key: 'popularity',
    icon: <TrendingUp className="w-5 h-5" />,
    name: 'Trending',
    description: 'Popular items',
    gradient: 'from-pink-500 to-rose-500',
    bgColor: 'bg-pink-50',
  },
  {
    key: 'price_segment',
    icon: <DollarSign className="w-5 h-5" />,
    name: 'Price Match',
    description: 'Budget aligned',
    gradient: 'from-slate-600 to-slate-700',
    bgColor: 'bg-slate-100',
  },
];

export function ModelSelector({ selectedModel, onSelectModel }: ModelSelectorProps) {
  const currentModel = selectedModel || 'auto';

  return (
    <div className="card p-5">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-800">Algorithm</h2>
          <p className="text-xs text-slate-500">Choose recommendation model</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {models.map((model) => {
          const isSelected = currentModel === model.key;
          return (
            <button
              key={model.key}
              onClick={() => onSelectModel(model.key === 'auto' ? null : model.key)}
              className={`relative group p-3 rounded-xl text-left transition-all duration-200 border-2 ${
                isSelected
                  ? `${model.bgColor} border-transparent ring-2 ring-offset-1 ring-slate-400`
                  : 'bg-white border-slate-100 hover:border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-start gap-2.5">
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${model.gradient} flex items-center justify-center flex-shrink-0 shadow-sm`}>
                  <span className="text-white">{model.icon}</span>
                </div>
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-sm text-slate-800 truncate">{model.name}</div>
                  <div className="text-[11px] text-slate-500 truncate">{model.description}</div>
                </div>
              </div>
              {isSelected && (
                <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center shadow-sm">
                  <Check className="w-3 h-3 text-white" strokeWidth={3} />
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
