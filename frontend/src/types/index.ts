export type UserType = 'loyal' | 'new' | 'unknown';

export type SpendingSegment = 'small' | 'low_average' | 'average' | 'high';

export type ModelType =
  | 'item_cf'
  | 'matrix_factorization'
  | 'als'
  | 'popularity'
  | 'price_segment'
  | 'hybrid'
  | 'blend';

export interface RecommendationItem {
  item_id: string;
  relevance_score: number;
  confidence_score: number;
  item_price: number | null;
  recommendation_reason: string;
  model_source: string;
}

export interface UserInfo {
  user_id: string;
  user_type: UserType;
  spending_segment: SpendingSegment | null;
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
  relevance_score: number;
  confidence_score: number;
  item_price: number | null;
}

export interface SimilarItemsResponse {
  item_id: string;
  similar_items: SimilarItem[];
  processing_time_ms: number;
}

export interface PopularItem {
  item_id: string;
  item_price: number | null;
  purchase_count: number;
  unique_buyers: number;
  popularity_score: number;
  confidence_score?: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: boolean;
  model_version: string | null;
  timestamp: string;
}

export interface ModelInfo {
  type: ModelType;
  name: string;
  description: string;
  best_for: string;
  weights?: Record<string, number>;
}

export interface UsersResponse {
  loyal_count: number;
  new_count: number;
  users: {
    loyal: string[];
    new: string[];
  };
}

export interface TuningConfig {
  candidate_pool_size: number;
  model_weights: {
    als: number;
    item_cf: number;
    popularity: number;
  };
  upsell: {
    enabled: boolean;
    factor: number;
    price_boost_weight: number;
  };
  user_segments: Record<string, number>;
  repurchase_cycle: {
    enabled: boolean;
    default_cycle_days: number;
  };
}
