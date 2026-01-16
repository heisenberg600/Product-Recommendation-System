import { Sparkles, Activity, Cpu } from 'lucide-react';
import { useHealth } from '../hooks/useRecommendations';

export function Header() {
  const { data: health, isLoading } = useHealth();

  return (
    <header className="bg-white border-b border-slate-200/80 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/25">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900">
                RecoAI
              </h1>
              <p className="text-xs text-slate-500 -mt-0.5">Product Recommendations</p>
            </div>
          </div>

          {/* Status */}
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-50 text-sm">
              <Cpu className="w-4 h-4 text-slate-500" />
              <span className="text-slate-600 font-medium">5 Models</span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-50">
              <Activity
                className={`w-4 h-4 ${
                  isLoading
                    ? 'text-slate-400 animate-pulse'
                    : health?.status === 'healthy'
                    ? 'text-emerald-500'
                    : 'text-amber-500'
                }`}
              />
              <span
                className={`text-sm font-medium ${
                  isLoading
                    ? 'text-slate-400'
                    : health?.status === 'healthy'
                    ? 'text-emerald-600'
                    : 'text-amber-600'
                }`}
              >
                {isLoading ? 'Connecting...' : health?.status === 'healthy' ? 'Online' : 'Starting'}
              </span>
              {!isLoading && health?.status === 'healthy' && (
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
