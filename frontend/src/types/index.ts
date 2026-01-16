export type UserType = 'loyal' | 'new' | 'unknown';

export type ModelType =
  | 'item_cf'
  | 'matrix_factorization'
  | 'popularity'
  | 'price_segment'
  | 'hybrid';

export interface RecommendationItem {
  item_id: string;
  relevance_score: number;
  confidence: number;
  item_price: number | null;
  recommendation_reason: string;
  model_used: ModelType;
}

export interface UserInfo {
  user_id: string;
  user_type: UserType;
  total_purchases: number;
  unique_items: number;
  avg_item_price: number | null;
  last_purchase_date: string | null;
}

export interface RecommendationResponse {
  user_id: string;
  user_info: UserInfo;
  recommendations: RecommendationItem[];
  primary_model: ModelType;
  fallback_used: boolean;
  processing_time_ms: number;
  generated_at: string;
}

export interface SimilarItem {
  item_id: string;
  similarity_score: number;
  item_price: number | null;
  purchase_count: number | null;
}

export interface PopularItem {
  item_id: string;
  item_price: number | null;
  purchase_count: number;
  unique_buyers: number;
  popularity_score: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: boolean;
  data_loaded: boolean;
  timestamp: string;
}

export interface ModelInfo {
  type: ModelType;
  name: string;
  description: string;
  best_for: string;
}

export interface UsersResponse {
  loyal_count: number;
  new_count: number;
  users: {
    loyal: string[];
    new: string[];
  };
}
