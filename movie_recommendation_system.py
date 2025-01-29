import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
import os
from typing import List, Dict, Optional, Tuple, Union
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieRecommender:
    """A movie recommendation system using content-based filtering and popularity metrics."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the recommendation system.
        
        Args:
            model_path: Optional path to load pre-trained models
        """
        self.movies: Optional[pd.DataFrame] = None
        self.ratings: Optional[pd.DataFrame] = None
        self.content_sim: Optional[np.ndarray] = None
        self.popularity_model: Optional[pd.DataFrame] = None
        
        if model_path:
            self._safe_load_models(model_path)
    
    def _validate_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
        """Validate the input dataframes.
        
        Args:
            movies_df: Movies dataframe
            ratings_df: Ratings dataframe
            
        Raises:
            ValueError: If data validation fails
        """
        required_movie_cols = {'movieId', 'title', 'genres'}
        required_rating_cols = {'userId', 'movieId', 'rating'}
        
        missing_movie_cols = required_movie_cols - set(movies_df.columns)
        missing_rating_cols = required_rating_cols - set(ratings_df.columns)
        
        if missing_movie_cols:
            raise ValueError(f"Movies dataframe missing columns: {missing_movie_cols}")
        if missing_rating_cols:
            raise ValueError(f"Ratings dataframe missing columns: {missing_rating_cols}")
            
        if movies_df.empty:
            raise ValueError("Movies dataframe is empty")
        if ratings_df.empty:
            raise ValueError("Ratings dataframe is empty")
            
        if not pd.api.types.is_numeric_dtype(ratings_df['rating']):
            raise ValueError("Ratings column must be numeric")
            
        invalid_ratings = ratings_df[~ratings_df['rating'].between(0.5, 5.0)]
        if not invalid_ratings.empty:
            warnings.warn(f"Found {len(invalid_ratings)} ratings outside valid range (0.5-5.0)")
    
    def load_data(self, movies_path: str, ratings_path: str) -> None:
        """Load and preprocess MovieLens dataset.
        
        Args:
            movies_path: Path to movies CSV file
            ratings_path: Path to ratings CSV file
            
        Raises:
            FileNotFoundError: If either file is not found
            ValueError: If data validation fails
        """
        logger.info("Loading datasets...")
        
        # Validate file paths
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        
        try:
            movies_df = pd.read_csv(movies_path)
            ratings_df = pd.read_csv(ratings_path)
            
            # Validate data
            self._validate_data(movies_df, ratings_df)
            
            # Clean and preprocess
            movies_df['genres'] = movies_df['genres'].fillna('')
            movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
            
            self.movies = movies_df
            self.ratings = ratings_df
            
            # Initialize models
            self._build_popularity_model()
            
            logger.info(f"Loaded {len(self.movies)} movies and {len(self.ratings)} ratings")
            
        except pd.errors.EmptyDataError:
            raise ValueError("One or both input files are empty")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _build_popularity_model(self) -> None:
        """Build the popularity-based recommendation model."""
        if self.ratings is None:
            raise ValueError("No ratings data available")
            
        try:
            # Calculate rating metrics
            rating_stats = self.ratings.groupby('movieId').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            
            rating_stats.columns = ['movieId', 'rating_count', 'rating_mean']
            
            # Normalize metrics
            rating_stats['count_norm'] = rating_stats['rating_count'] / rating_stats['rating_count'].max()
            rating_stats['mean_norm'] = rating_stats['rating_mean'] / 5.0
            
            # Calculate popularity score (weighted average of normalized count and mean)
            rating_stats['popularity_score'] = (
                rating_stats['count_norm'] * 0.6 +  # Weight count more heavily
                rating_stats['mean_norm'] * 0.4
            )
            
            self.popularity_model = rating_stats
            
        except Exception as e:
            logger.error(f"Error building popularity model: {str(e)}")
            raise
    
    def build_content_model(self) -> None:
        """Build content-based filtering model using TF-IDF."""
        logger.info("Building content-based model...")
        
        if self.movies is None:
            raise ValueError("No movie data available")
            
        try:
            # Create TF-IDF vectorizer
            tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                strip_accents='unicode'
            )
            
            # Create genre matrix
            genre_matrix = tfidf.fit_transform(self.movies['genres'])
            
            # Calculate similarity matrix
            self.content_sim = cosine_similarity(genre_matrix, genre_matrix)
            
            logger.info("Content-based model built successfully")
            
        except Exception as e:
            logger.error(f"Error building content model: {str(e)}")
            raise
    
    def recommend(self, 
                 user_id: Optional[int] = None, 
                 movie_title: Optional[str] = None, 
                 n: int = 10) -> pd.DataFrame:
        """Generate movie recommendations.
        
        Args:
            user_id: Optional user ID for personalized recommendations
            movie_title: Optional movie title for content-based recommendations
            n: Number of recommendations to return
            
        Returns:
            DataFrame with movie recommendations and scores
            
        Raises:
            ValueError: If neither user_id nor movie_title is provided
        """
        try:
            if n < 1:
                raise ValueError("Number of recommendations must be positive")
                
            if movie_title:
                return self._get_content_recommendations(movie_title, n)
            elif user_id:
                return self._get_user_recommendations(user_id, n)
            else:
                return self._get_popular_recommendations(n)
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._get_popular_recommendations(min(n, 10))  # Fallback to popular
    
    def _get_content_recommendations(self, movie_title: str, n: int) -> pd.DataFrame:
        """Get content-based recommendations for a movie."""
        if self.content_sim is None:
            raise ValueError("Content model not built")
            
        try:
            # Find movie index
            movie_matches = self.movies[self.movies['title'].str.lower() == movie_title.lower()]
            if movie_matches.empty:
                logger.warning(f"Movie not found: {movie_title}")
                return self._get_popular_recommendations(n)
            
            movie_idx = movie_matches.index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_sim[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Filter out the input movie and get top N
            sim_scores = [s for s in sim_scores if s[0] != movie_idx][:n]
            
            # Create recommendations dataframe
            recs = pd.DataFrame(sim_scores, columns=['index', 'similarity'])
            recs['title'] = recs['index'].apply(lambda x: self.movies.iloc[x]['title'])
            
            return recs[['title', 'similarity']].rename(columns={'similarity': 'score'})
            
        except Exception as e:
            logger.error(f"Error in content-based recommendation: {str(e)}")
            return self._get_popular_recommendations(n)
    
    def _get_user_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Get personalized recommendations for a user."""
        if self.content_sim is None:
            raise ValueError("Content model not built")
            
        try:
            # Get user's ratings
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            if user_ratings.empty:
                logger.warning(f"No ratings found for user {user_id}")
                return self._get_popular_recommendations(n)
            
            # Get highly rated movies
            liked_movies = user_ratings[user_ratings['rating'] >= 4.0]
            if liked_movies.empty:
                logger.warning(f"No high ratings found for user {user_id}")
                return self._get_popular_recommendations(n)
            
            # Collect recommendations from each liked movie
            all_recs = []
            for _, row in liked_movies.iterrows():
                movie_idx = self.movies[self.movies['movieId'] == row['movieId']].index
                if len(movie_idx) > 0:
                    sim_scores = list(enumerate(self.content_sim[movie_idx[0]]))
                    # Weight similarity by user's rating
                    sim_scores = [(idx, score * (row['rating'] / 5.0)) 
                                for idx, score in sim_scores]
                    all_recs.extend(sim_scores)
            
            # Aggregate and sort recommendations
            movie_scores = {}
            for idx, score in all_recs:
                if idx not in movie_scores:
                    movie_scores[idx] = score
                else:
                    # Take maximum similarity score
                    movie_scores[idx] = max(movie_scores[idx], score)
            
            # Convert to dataframe and sort
            recs = pd.DataFrame(list(movie_scores.items()), columns=['index', 'score'])
            recs['title'] = recs['index'].apply(lambda x: self.movies.iloc[x]['title'])
            
            # Remove movies the user has already rated
            rated_movies = set(user_ratings['movieId'])
            recs = recs[~recs['index'].isin(
                self.movies[self.movies['movieId'].isin(rated_movies)].index
            )]
            
            return recs.nlargest(n, 'score')[['title', 'score']]
            
        except Exception as e:
            logger.error(f"Error in user recommendation: {str(e)}")
            return self._get_popular_recommendations(n)
    
    def _get_popular_recommendations(self, n: int) -> pd.DataFrame:
        """Get recommendations based on popularity."""
        try:
            if self.popularity_model is None:
                raise ValueError("Popularity model not built")
                
            popular = self.popularity_model.merge(
                self.movies[['movieId', 'title']], 
                on='movieId'
            )
            
            return popular.nlargest(n, 'popularity_score')[
                ['title', 'popularity_score']
            ].rename(columns={'popularity_score': 'score'})
            
        except Exception as e:
            logger.error(f"Error in popularity recommendation: {str(e)}")
            # Return empty dataframe with correct columns
            return pd.DataFrame(columns=['title', 'score'])
    
    def _safe_load_models(self, path: str) -> None:
        """Safely load pretrained models from disk."""
        try:
            if not os.path.exists(path):
                logger.warning(f"Model file not found: {path}")
                return
                
            models = joblib.load(path)
            
            required_keys = {'content_sim', 'popularity_model'}
            if not all(key in models for key in required_keys):
                logger.warning("Model file missing required components")
                return
                
            self.content_sim = models['content_sim']
            self.popularity_model = models['popularity_model']
            
            logger.info(f"Models loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def save_models(self, path: str) -> None:
        """Save trained models to disk.
        
        Args:
            path: Path to save the models
            
        Raises:
            ValueError: If models haven't been built
        """
        if self.content_sim is None or self.popularity_model is None:
            raise ValueError("Models not built")
            
        try:
            model_data = {
                'content_sim': self.content_sim,
                'popularity_model': self.popularity_model
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Models saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

def main():
    """Example usage of the recommendation system."""
    try:
        # Initialize system
        recommender = MovieRecommender()
        
        # Load data
        recommender.load_data('movies.csv', 'ratings.csv')
        
        # Build models
        recommender.build_content_model()
        
        # Save models
        recommender.save_models('movie_recommender_models.pkl')
        
        # Generate different types of recommendations
        print("\nContent-based recommendations for 'Toy Story (1995)':")
        print(recommender.recommend(movie_title='Toy Story (1995)'))
        
        print("\nPersonalized recommendations for user 1:")
        print(recommender.recommend(user_id=1))
        
        print("\nPopular movie recommendations:")
        print(recommender.recommend())
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
