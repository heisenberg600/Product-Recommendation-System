import { Sparkles, Activity, Cpu, Zap } from 'lucide-react';
import { useHealth } from '../hooks/useRecommendations';

export function Header() {
  const { data: health, isLoading } = useHealth();

  return (
    <header className="bg-white/80 backdrop-blur-lg border-b border-slate-200/50 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/30">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div className="absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full bg-gradient-to-br from-emerald-400 to-green-500 border-2 border-white flex items-center justify-center">
                <Zap className="w-2.5 h-2.5 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                RecoAI
              </h1>
              <p className="text-xs text-slate-500 -mt-0.5 font-medium">AI-Powered Recommendations</p>
            </div>
          </div>

          {/* Status */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-slate-50 to-slate-100 border border-slate-200">
              <Cpu className="w-4 h-4 text-indigo-500" />
              <span className="text-slate-700 font-semibold text-sm">5 Models</span>
            </div>

            <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${
              isLoading
                ? 'bg-slate-50 border-slate-200'
                : health?.status === 'healthy'
                ? 'bg-gradient-to-r from-emerald-50 to-green-50 border-emerald-200'
                : 'bg-gradient-to-r from-amber-50 to-orange-50 border-amber-200'
            }`}>
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
                className={`text-sm font-semibold ${
                  isLoading
                    ? 'text-slate-400'
                    : health?.status === 'healthy'
                    ? 'text-emerald-700'
                    : 'text-amber-700'
                }`}
              >
                {isLoading ? 'Connecting...' : health?.status === 'healthy' ? 'Online' : 'Starting'}
              </span>
              {!isLoading && health?.status === 'healthy' && (
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
