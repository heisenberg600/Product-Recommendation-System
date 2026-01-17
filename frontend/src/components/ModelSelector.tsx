import { Brain, Layers, Zap, Check } from 'lucide-react';
import type { ModelType } from '../types';

interface ModelSelectorProps {
  selectedModel: ModelType | null;
  onSelectModel: (model: ModelType | null) => void;
}

const models: { key: ModelType; icon: React.ReactNode; name: string; description: string; gradient: string; bgColor: string }[] = [
  {
    key: 'hybrid',
    icon: <Layers className="w-5 h-5" />,
    name: 'Hybrid',
    description: 'Best of all models',
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
];

export function ModelSelector({ selectedModel, onSelectModel }: ModelSelectorProps) {
  // Default to hybrid if nothing selected
  const currentModel = selectedModel || 'hybrid';

  return (
    <div className="card p-5 border-0 shadow-xl shadow-slate-200/50">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
          <Layers className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-800">Algorithm</h2>
          <p className="text-xs text-slate-500">Choose recommendation model</p>
        </div>
      </div>

      <div className="space-y-2">
        {models.map((model) => {
          const isSelected = currentModel === model.key;
          return (
            <button
              key={model.key}
              onClick={() => onSelectModel(model.key === 'hybrid' ? null : model.key)}
              className={`relative group w-full p-3 rounded-xl text-left transition-all duration-200 border-2 ${
                isSelected
                  ? `${model.bgColor} border-transparent ring-2 ring-offset-1 ring-slate-400`
                  : 'bg-white border-slate-100 hover:border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${model.gradient} flex items-center justify-center flex-shrink-0 shadow-sm`}>
                  <span className="text-white">{model.icon}</span>
                </div>
                <div className="min-w-0 flex-1">
                  <div className="font-semibold text-sm text-slate-800">{model.name}</div>
                  <div className="text-xs text-slate-500">{model.description}</div>
                </div>
                {isSelected && (
                  <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center shadow-sm">
                    <Check className="w-3.5 h-3.5 text-white" strokeWidth={3} />
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
