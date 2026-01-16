import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { ModelType } from '../types';

export function useRecommendations(
  userId: string | null,
  options?: {
    n?: number;
    model?: ModelType;
    excludePurchased?: boolean;
    priceMin?: number;
    priceMax?: number;
  }
) {
  return useQuery({
    queryKey: ['recommendations', userId, options],
    queryFn: () => apiClient.getRecommendations(userId!, options),
    enabled: !!userId,
    staleTime: 30000,
  });
}

export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: apiClient.getUsers,
    staleTime: 60000,
  });
}

export function usePopularItems(n: number = 10) {
  return useQuery({
    queryKey: ['popular-items', n],
    queryFn: () => apiClient.getPopularItems(n),
    staleTime: 60000,
  });
}

export function useSimilarItems(itemId: string | null, n: number = 10) {
  return useQuery({
    queryKey: ['similar-items', itemId, n],
    queryFn: () => apiClient.getSimilarItems(itemId!, n),
    enabled: !!itemId,
    staleTime: 60000,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: apiClient.getModels,
    staleTime: 300000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: apiClient.getHealth,
    refetchInterval: 10000,
    staleTime: 5000,
  });
}

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: apiClient.getStats,
    staleTime: 30000,
  });
}
