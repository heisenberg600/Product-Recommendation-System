import axios from 'axios';
import type {
  RecommendationResponse,
  HealthResponse,
  ModelInfo,
  UsersResponse,
  PopularItem,
  SimilarItem,
  ModelType,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiClient = {
  // Health check
  getHealth: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },

  // Get recommendations for a user
  getRecommendations: async (
    userId: string,
    options?: {
      n?: number;
      model?: ModelType;
      excludePurchased?: boolean;
      priceMin?: number;
      priceMax?: number;
    }
  ): Promise<RecommendationResponse> => {
    const params = new URLSearchParams();
    if (options?.n) params.append('n', options.n.toString());
    if (options?.model) params.append('model', options.model);
    if (options?.excludePurchased !== undefined)
      params.append('exclude_purchased', options.excludePurchased.toString());
    if (options?.priceMin) params.append('price_min', options.priceMin.toString());
    if (options?.priceMax) params.append('price_max', options.priceMax.toString());

    const response = await api.get<RecommendationResponse>(
      `/recommendations/${userId}?${params.toString()}`
    );
    return response.data;
  },

  // Get similar items
  getSimilarItems: async (
    itemId: string,
    n: number = 10
  ): Promise<{ item_id: string; similar_items: SimilarItem[] }> => {
    const response = await api.get(`/items/${itemId}/similar?n=${n}`);
    return response.data;
  },

  // Get popular items
  getPopularItems: async (n: number = 10): Promise<{ count: number; items: PopularItem[] }> => {
    const response = await api.get(`/items/popular?n=${n}`);
    return response.data;
  },

  // Get all users
  getUsers: async (): Promise<UsersResponse> => {
    const response = await api.get<UsersResponse>('/users');
    return response.data;
  },

  // Get available models
  getModels: async (): Promise<{ models: ModelInfo[] }> => {
    const response = await api.get('/models');
    return response.data;
  },

  // Get system statistics
  getStats: async (): Promise<Record<string, unknown>> => {
    const response = await api.get('/stats');
    return response.data;
  },
};
