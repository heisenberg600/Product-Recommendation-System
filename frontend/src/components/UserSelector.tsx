import { useState } from 'react';
import { Search, Users, Crown, UserPlus, Check, UserX } from 'lucide-react';
import { useUsers } from '../hooks/useRecommendations';

interface UserSelectorProps {
  selectedUser: string | null;
  onSelectUser: (userId: string | null, isLoyal?: boolean) => void;
  onSelectAnonymous?: () => void;
  isAnonymous?: boolean;
}

export function UserSelector({ selectedUser, onSelectUser, onSelectAnonymous, isAnonymous }: UserSelectorProps) {
  const { data: usersData, isLoading } = useUsers();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'loyal' | 'new'>('loyal');

  const users = usersData?.users[activeTab] || [];
  const filteredUsers = users.filter((u) =>
    u.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="card p-5 border-0 shadow-xl shadow-slate-200/50">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
          <Users className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-800">Customers</h2>
          <p className="text-xs text-slate-500">
            {usersData
              ? `${usersData.loyal_count} loyal, ${usersData.new_count} new`
              : 'Loading...'}
          </p>
        </div>
      </div>

      {/* Anonymous User Option */}
      {onSelectAnonymous && (
        <button
          onClick={() => {
            onSelectAnonymous();
            onSelectUser(null);
          }}
          className={`w-full flex items-center gap-3 px-4 py-3 mb-4 rounded-xl transition-all ${
            isAnonymous
              ? 'bg-gradient-to-r from-violet-500 to-purple-500 text-white shadow-lg shadow-violet-500/30'
              : 'bg-gradient-to-r from-violet-50 to-purple-50 text-violet-700 hover:from-violet-100 hover:to-purple-100 border border-violet-200'
          }`}
        >
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isAnonymous ? 'bg-white/20' : 'bg-violet-100'}`}>
            <UserX className={`w-4 h-4 ${isAnonymous ? 'text-white' : 'text-violet-600'}`} />
          </div>
          <div className="text-left">
            <div className="font-semibold text-sm">New Visitor</div>
            <div className={`text-xs ${isAnonymous ? 'text-violet-200' : 'text-violet-500'}`}>No purchase history</div>
          </div>
          {isAnonymous && (
            <div className="ml-auto w-5 h-5 rounded-full bg-white flex items-center justify-center">
              <Check className="w-3 h-3 text-violet-600" strokeWidth={3} />
            </div>
          )}
        </button>
      )}

      {/* Tab Switcher */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setActiveTab('loyal')}
          className={`flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
            activeTab === 'loyal'
              ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-md shadow-amber-500/25'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          <Crown className="w-4 h-4" />
          <span>Loyal</span>
          <span className={`text-xs px-1.5 py-0.5 rounded ${activeTab === 'loyal' ? 'bg-white/20' : 'bg-slate-200'}`}>
            {usersData?.loyal_count || 0}
          </span>
        </button>
        <button
          onClick={() => setActiveTab('new')}
          className={`flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
            activeTab === 'new'
              ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-md shadow-emerald-500/25'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          <UserPlus className="w-4 h-4" />
          <span>New</span>
          <span className={`text-xs px-1.5 py-0.5 rounded ${activeTab === 'new' ? 'bg-white/20' : 'bg-slate-200'}`}>
            {usersData?.new_count || 0}
          </span>
        </button>
      </div>

      {/* Search */}
      <div className="relative mb-4">
        <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
        <input
          type="text"
          placeholder="Search users..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full py-2.5 pl-11 pr-4 rounded-lg border-2 border-slate-200 bg-white text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
        />
      </div>

      {/* User List */}
      <div className="max-h-56 overflow-y-auto space-y-1.5">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="skeleton h-11 rounded-lg" />
            ))}
          </div>
        ) : filteredUsers.length === 0 ? (
          <div className="text-center py-8 text-slate-400 text-sm">
            No users found
          </div>
        ) : (
          filteredUsers.map((userId) => {
            const isSelected = selectedUser === userId;
            return (
              <button
                key={userId}
                onClick={() => onSelectUser(userId, activeTab === 'loyal')}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg transition-all ${
                  isSelected
                    ? 'bg-blue-50 ring-2 ring-blue-500 ring-offset-1'
                    : 'bg-slate-50 hover:bg-slate-100'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                      activeTab === 'loyal'
                        ? 'bg-amber-100 text-amber-600'
                        : 'bg-emerald-100 text-emerald-600'
                    }`}
                  >
                    {activeTab === 'loyal' ? (
                      <Crown className="w-3.5 h-3.5" />
                    ) : (
                      <UserPlus className="w-3.5 h-3.5" />
                    )}
                  </div>
                  <span className="font-mono text-sm text-slate-700">{userId}</span>
                </div>
                {isSelected && (
                  <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center">
                    <Check className="w-3 h-3 text-white" strokeWidth={3} />
                  </div>
                )}
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
