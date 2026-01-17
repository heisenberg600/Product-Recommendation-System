import axios from 'axios';
import type {
  RecommendationResponse,
  HealthResponse,
  ModelInfo,
  UsersResponse,
  PopularItem,
  SimilarItemsResponse,
  ModelType,
  TuningConfig,
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
      model?: ModelType | string;
      timestamp?: string;
      excludePurchased?: boolean;
      priceMin?: number;
      priceMax?: number;
    }
  ): Promise<RecommendationResponse> => {
    const params = new URLSearchParams();
    if (options?.n) params.append('n', options.n.toString());
    if (options?.model) params.append('model', options.model);
    if (options?.timestamp) params.append('timestamp', options.timestamp);
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
    n: number = 5
  ): Promise<SimilarItemsResponse> => {
    const response = await api.get<SimilarItemsResponse>(`/items/${itemId}/similar?n=${n}`);
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
  getModels: async (): Promise<{ models: ModelInfo[]; architecture: Record<string, string> }> => {
    const response = await api.get('/models');
    return response.data;
  },

  // Get tuning configuration
  getConfig: async (): Promise<{ config: TuningConfig; description: Record<string, string> }> => {
    const response = await api.get('/config');
    return response.data;
  },

  // Get system statistics
  getStats: async (): Promise<Record<string, unknown>> => {
    const response = await api.get('/stats');
    return response.data;
  },

  // Get recommendations for anonymous users
  getAnonymousRecommendations: async (n: number = 5, priceMax?: number): Promise<{
    user_type: string;
    recommendations: Array<{
      item_id: string;
      relevance_score: number;
      confidence_score: number;
      item_price: number;
      recommendation_reason: string;
      model_source: string;
    }>;
    primary_model: string;
    message: string;
  }> => {
    const params = new URLSearchParams();
    params.append('n', n.toString());
    if (priceMax) params.append('price_max', priceMax.toString());
    const response = await api.get(`/recommendations/anonymous?${params.toString()}`);
    return response.data;
  },

  // Get recommendations based on custom item list
  getCustomRecommendations: async (
    items: string[],
    n: number = 5,
    timestamp?: string
  ): Promise<{
    input_items: string[];
    timestamp: string | null;
    recommendations: Array<{
      item_id: string;
      relevance_score: number;
      confidence_score: number;
      item_price: number;
      recommendation_reason: string;
      model_source: string;
    }>;
    primary_model: string;
    message: string;
  }> => {
    const params = new URLSearchParams();
    params.append('n', n.toString());
    if (timestamp) params.append('timestamp', timestamp);
    const response = await api.post(`/recommendations/custom?${params.toString()}`, items);
    return response.data;
  },

  // Get user profile
  getUserProfile: async (userId: string): Promise<{
    user_id: string;
    user_info: Record<string, unknown>;
    profile: Record<string, unknown> | null;
    is_known_user: boolean;
  }> => {
    const response = await api.get(`/users/${userId}/profile`);
    return response.data;
  },
};
