import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, ResNet50V2, DenseNet121
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Precision, Recall
from tensorflow.keras import mixed_precision

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retinal_cnn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management for team deployment"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration if config file is missing"""
        return {
            'data': {
                'img_size': [512, 512],
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42
            },
            'model': {
                'base_model': 'EfficientNetB3',
                'num_classes': 5,
                'dropout_rate': 0.5
            },
            'training': {
                'batch_size': 16,
                'class_weight_strategy': 'balanced'
            }
        }

class DataManager:
    """Enhanced data management with better error handling"""
    
    def __init__(self, config):
        self.config = config['data']
        self.class_labels = {
            0: "No DR", 1: "Mild DR", 2: "Moderate DR",
            3: "Severe DR", 4: "Proliferative DR"
        }
        
    def validate_data_paths(self):
        """Validate that all required data paths exist"""
        base_path = Path(self.config['base_path'])
        csv_path = Path(self.config['csv_path'])
        
        if not base_path.exists():
            raise FileNotFoundError(f"Base path not found: {base_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        logger.info("Data paths validated successfully")
        return True
    
    def load_data(self):
        """Load and validate image data"""
        self.validate_data_paths()
        
        # Load CSV
        df = pd.read_csv(self.config['csv_path'])
        logger.info(f"Loaded CSV with {len(df)} records")
        
        # Find image directories
        base_path = Path(self.config['base_path'])
        folder_dirs = sorted([d for d in base_path.iterdir() 
                            if d.is_dir() and d.name.startswith('Folder')],
                           key=lambda x: int(x.name.replace('Folder', '')))
        
        logger.info(f"Found folders: {[f.name for f in folder_dirs]}")
        
        # Match images to labels
        image_paths, labels = [], []
        id_to_label = dict(zip(df['id_code'], df['diagnosis']))
        
        for folder in folder_dirs:
            image_files = list(folder.glob('*.png'))
            for img_path in image_files:
                patient_id = img_path.stem
                if patient_id in id_to_label:
                    image_paths.append(str(img_path))
                    labels.append(id_to_label[patient_id])
        
        logger.info(f"Successfully matched {len(image_paths)} images")
        
        # Create summary DataFrame
        self.data_summary = pd.DataFrame({
            'image_path': image_paths,
            'diagnosis': labels
        })
        
        self._display_class_distribution()
        return image_paths, labels
    
    def _display_class_distribution(self):
        """Display class distribution with logging"""
        dist = self.data_summary['diagnosis'].value_counts().sort_index()
        total = len(self.data_summary)
        
        logger.info("Class Distribution:")
        for diagnosis, count in dist.items():
            percentage = (count / total) * 100
            label_name = self.class_labels.get(diagnosis, f"Unknown-{diagnosis}")
            logger.info(f"  {diagnosis}: {label_name:<15} {count:4d} ({percentage:5.1f}%)")
        
        # Check for class imbalance
        max_count, min_count = dist.max(), dist.min()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 5:
            logger.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")

class AugmentationManager:
    """Centralized augmentation management"""
    
    def __init__(self, config):
        self.config = config.get('augmentation', {})
        
    def create_train_augmentation(self):
        """Create training augmentation pipeline"""
        return A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=self.config.get('geometric', {}).get('random_rotate90', 0.5)),
            A.HorizontalFlip(p=self.config.get('geometric', {}).get('flip', 0.5)),
            A.VerticalFlip(p=self.config.get('geometric', {}).get('flip', 0.5)),
            A.Transpose(p=self.config.get('geometric', {}).get('transpose', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, 
                p=self.config.get('geometric', {}).get('shift_scale_rotate', {}).get('probability', 0.8)
            ),
            
            # Optical distortions
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, 
                               p=self.config.get('optical', {}).get('optical_distortion', 0.3)),
            A.GridDistortion(num_steps=5, distort_limit=0.1, 
                            p=self.config.get('optical', {}).get('grid_distortion', 0.3)),
            
            # Color adjustments
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, 
                                     p=self.config.get('color', {}).get('brightness_contrast', 0.8)),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, 
                               p=self.config.get('color', {}).get('hue_saturation', 0.7)),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), 
                   p=self.config.get('color', {}).get('clahe', 0.5)),
            
            # Noise and effects
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(1, 3)),
                A.MotionBlur(blur_limit=3),
            ], p=self.config.get('noise', {}).get('probability', 0.3)),
            
            # Cutout
            A.CoarseDropout(
                max_holes=self.config.get('cutout', {}).get('max_holes', 8),
                max_height=self.config.get('cutout', {}).get('max_height', 32),
                max_width=self.config.get('cutout', {}).get('max_width', 32),
                min_holes=1, fill_value=0, 
                p=self.config.get('cutout', {}).get('coarse_dropout', 0.3)
            ),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class ModelFactory:
    """Factory pattern for model creation"""
    
    def __init__(self, config):
        self.config = config
        
    @staticmethod
    def get_base_model(model_name, input_shape):
        """Get pretrained base model"""
        models_dict = {
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB3': EfficientNetB3,
            'ResNet50V2': ResNet50V2,
            'DenseNet121': DenseNet121
        }
        
        if model_name not in models_dict:
            raise ValueError(f"Unsupported model: {model_name}")
            
        return models_dict[model_name](
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
    
    def create_focal_loss(self, alpha=0.25, gamma=2.0):
        """Create focal loss for class imbalance"""
        def focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
            y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)
            
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
            p_t = tf.where(tf.equal(y_true_one_hot, 1), y_pred, 1 - y_pred)
            alpha_t = tf.ones_like(p_t) * alpha
            alpha_t = tf.where(tf.equal(y_true_one_hot, 1), alpha_t, 1 - alpha_t)
            
            focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
            focal_loss = focal_weight * ce_loss
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss
    
    def create_model(self, img_size=(512, 512)):
        """Create enhanced model with attention mechanism"""
        model_config = self.config['model']
        
        # Get base model
        base = self.get_base_model(
            model_config['base_model'], 
            (*img_size, 3)
        )
        base.trainable = False
        
        # Build model
        inputs = layers.Input(shape=(*img_size, 3))
        x = base(inputs)
        
        # Global Average Pooling + Attention
        x = layers.GlobalAveragePooling2D()(x)
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        # Classification head
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(model_config['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(model_config['dropout_rate'] * 0.8)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(model_config['dropout_rate'] * 0.6)(x)
        
        # Output layer
        outputs = layers.Dense(
            model_config['num_classes'], 
            activation='softmax', 
            dtype='float32', 
            name='predictions'
        )(x)
        
        model = models.Model(inputs, outputs)
        
        # Compile model
        loss_config = self.config.get('loss', {})
        if loss_config.get('type') == 'focal':
            loss_fn = self.create_focal_loss(
                alpha=loss_config.get('focal_alpha', 0.25),
                gamma=loss_config.get('focal_gamma', 2.0)
            )
        else:
            loss_fn = SparseCategoricalCrossentropy()
        
        model.compile(
            optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=loss_fn,
            metrics=[
                SparseCategoricalAccuracy(name='accuracy'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )
        
        return model, base

class EnhancedRetinalCNNv2:
    """Team-ready version of Enhanced Retinal CNN"""
    
    def __init__(self, config_path="config.yaml"):
        self.config = ConfigManager(config_path).config
        self.data_manager = DataManager(self.config)
        self.augmentation_manager = AugmentationManager(self.config)
        self.model_factory = ModelFactory(self.config)
        
        # Initialize components
        self.model = None
        self.base_model = None
        self.history = None
        self.class_weights = None
        
        # Setup directories
        self._setup_directories()
        
        # Configure GPU if available
        self._configure_gpu()
        
        logger.info("EnhancedRetinalCNNv2 initialized successfully")
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs_to_create = ['models', 'logs', 'results']
        for dir_name in dirs_to_create:
            Path(dir_name).mkdir(exist_ok=True)
    
    def _configure_gpu(self):
        """Configure GPU settings"""
        if self.config.get('hardware', {}).get('mixed_precision', True):
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled")
        
        # Enable GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
    
    def load_data(self):
        """Load and prepare data"""
        return self.data_manager.load_data()
    
    def compute_class_weights(self, labels, strategy='balanced'):
        """Compute class weights for imbalanced data"""
        unique_classes = np.unique(labels)
        
        if strategy == 'balanced':
            class_weights = compute_class_weight(
                'balanced', classes=unique_classes, y=labels
            )
        elif strategy == 'sqrt':
            class_counts = np.bincount(labels)
            class_weights = 1.0 / np.sqrt(class_counts[unique_classes])
            class_weights = class_weights / class_weights.sum() * len(unique_classes)
        
        self.class_weights = dict(zip(unique_classes, class_weights))
        
        logger.info(f"Class weights computed using '{strategy}' strategy:")
        for class_id, weight in self.class_weights.items():
            logger.info(f"  Class {class_id}: {weight:.3f}")
        
        return self.class_weights
    
    def preprocess_image(self, image_path, augment=False):
        """Preprocess single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            img_size = tuple(self.config['data']['img_size'])
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply augmentation
            if augment:
                train_aug = self.augmentation_manager.create_train_augmentation()
                augmented = train_aug(image=image)
                image = augmented['image']
            else:
                image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            img_size = tuple(self.config['data']['img_size'])
            return np.zeros((*img_size, 3), dtype=np.float32)
    
    def create_data_generator(self, image_paths, labels, batch_size=16, augment=False):
        """Create simpler data generator without variable batch sizes"""
        def generator():
            while True:
                # Always use same order for validation, shuffle for training
                indices = np.arange(len(image_paths))
                if augment:  # Training data
                    np.random.shuffle(indices)
                
                for start in range(0, (len(indices) // batch_size) * batch_size, batch_size):
                    batch_indices = indices[start:start + batch_size]
                    batch_images = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        img = self.preprocess_image(image_paths[idx], augment=augment)
                        batch_images.append(img)
                        batch_labels.append(labels[idx])
                    
                    yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.int32)
        
        return generator()
    
    def create_model(self):
        """Create and compile model"""
        img_size = tuple(self.config['data']['img_size'])
        self.model, self.base_model = self.model_factory.create_model(img_size)
        logger.info(f"Model created: {self.config['model']['base_model']}")
        return self.model
    
    def create_callbacks(self, model_name):
        """Create training callbacks"""
        callbacks_config = self.config.get('callbacks', {})
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor=callbacks_config.get('early_stopping', {}).get('monitor', 'val_accuracy'),
                patience=callbacks_config.get('early_stopping', {}).get('patience', 20),
                restore_best_weights=True,
                min_delta=callbacks_config.get('early_stopping', {}).get('min_delta', 0.001),
                verbose=1
            ),
            
            callbacks.ReduceLROnPlateau(
                monitor=callbacks_config.get('reduce_lr', {}).get('monitor', 'val_loss'),
                factor=callbacks_config.get('reduce_lr', {}).get('factor', 0.2),
                patience=callbacks_config.get('reduce_lr', {}).get('patience', 8),
                min_lr=callbacks_config.get('reduce_lr', {}).get('min_lr', 1e-7),
                verbose=1
            ),
            
            callbacks.ModelCheckpoint(
                filepath=f'models/{model_name}_best.h5',
                monitor=callbacks_config.get('model_checkpoint', {}).get('monitor', 'val_accuracy'),
                save_best_only=callbacks_config.get('model_checkpoint', {}).get('save_best_only', True),
                save_weights_only=False,
                verbose=1
            ),
            
            callbacks.CSVLogger(f'logs/{model_name}_training_log.csv')
        ]
        
        return callback_list
    
    def train_progressive(self, train_gen, val_gen, train_steps, val_steps):
        """Progressive training with unfreezing"""
        training_config = self.config['training']
        
        # Stage 1: Frozen base
        logger.info("Stage 1: Training with frozen base model")
        stage1_config = training_config['stage1']
        
        history1 = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=stage1_config['epochs'],
            callbacks=self.create_callbacks("stage1"),
            verbose=1
        )
        
        # Stage 2: Partial unfreezing
        logger.info("Stage 2: Partial unfreezing")
        stage2_config = training_config['stage2']
        
        # Unfreeze last layers
        unfreeze_layers = stage2_config.get('unfreeze_layers', 20)
        for layer in self.base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=AdamW(learning_rate=stage2_config['learning_rate'], weight_decay=1e-4),
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        history2 = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=stage2_config['epochs'],
            callbacks=self.create_callbacks("stage2"),
            verbose=1
        )
        
        # Stage 3: Full fine-tuning
        if stage2_config.get('full_unfreezing', True):
            logger.info("Stage 3: Full fine-tuning")
            stage3_config = training_config['stage3']
            
            self.base_model.trainable = True
            
            self.model.compile(
                optimizer=AdamW(learning_rate=stage3_config['learning_rate'], weight_decay=1e-4),
                loss=self.model.loss,
                metrics=self.model.metrics
            )
            
            history3 = self.model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                validation_data=val_gen,
                validation_steps=val_steps,
                epochs=stage3_config['epochs'],
                callbacks=self.create_callbacks("stage3"),
                verbose=1
            )
            
            # Combine histories
            combined_history = {}
            for key in history1.history.keys():
                combined_history[key] = (
                    history1.history[key] + 
                    history2.history[key] + 
                    history3.history[key]
                )
        else:
            # Just combine first two stages
            combined_history = {}
            for key in history1.history.keys():
                combined_history[key] = history1.history[key] + history2.history[key]
        
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        self.history = CombinedHistory(combined_history)
        logger.info("Progressive training completed")
        return self.history
    
    def evaluate_model(self, test_images, test_labels):
        """Comprehensive model evaluation"""
        logger.info("Starting model evaluation")
        
        # Get predictions
        y_pred_proba = self.model.predict(test_images, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        target_names = [f"Class_{i}" for i in range(len(np.unique(test_labels)))]
        report = classification_report(test_labels, y_pred, target_names=target_names)
        logger.info(f"Classification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, y_pred)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == test_labels)
        weighted_f1 = f1_score(test_labels, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")
        
        return results
    
    def save_results(self, results, filename=None):
        """Save evaluation results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/evaluation_results_{timestamp}.yaml"
        
        # Convert numpy arrays to lists for YAML serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False)
        
        logger.info(f"Results saved to {filename}")

# Main training pipeline function
def run_team_training_pipeline(config_path="config.yaml"):
    """Main training pipeline for team deployment"""
    
    logger.info("="*60)
    logger.info("STARTING ENHANCED RETINAL CNN TRAINING PIPELINE")
    logger.info("="*60)
    
    try:
        # Initialize system
        cnn = EnhancedRetinalCNNv2(config_path)
        
        # Load data
        image_paths, labels = cnn.load_data()
        
        # Compute class weights
        cnn.compute_class_weights(labels, strategy=cnn.config['training']['class_weight_strategy'])
        
        # Split data
        data_config = cnn.config['data']
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, 
            test_size=data_config['test_size'], 
            stratify=labels, 
            random_state=data_config['random_state']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=data_config['val_size'], 
            stratify=y_train, 
            random_state=data_config['random_state']
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # Create model
        model = cnn.create_model()
        
        # Create data generators
        batch_size = cnn.config['training']['batch_size']
        train_gen = cnn.create_data_generator(X_train, y_train, batch_size, augment=True)
        val_gen = cnn.create_data_generator(X_val, y_val, batch_size, augment=False)
        
        # Calculate steps
        train_steps = len(X_train) // batch_size
        val_steps = len(X_val) // batch_size
        
        # Train model with progressive unfreezing
        history = cnn.train_progressive(train_gen, val_gen, train_steps, val_steps)
        
        # Evaluate on test set
        logger.info("Preparing test data for evaluation...")
        test_images = np.array([cnn.preprocess_image(path) for path in X_test])
        results = cnn.evaluate_model(test_images, y_test)
        
        # Save results
        cnn.save_results(results)
        
        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Final Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Final Weighted F1-Score: {results['weighted_f1']:.4f}")
        
        return cnn, history, results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the training pipeline
    enhanced_cnn, history, results = run_team_training_pipeline("config.yaml")
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"📈 Final Weighted F1-Score: {results['weighted_f1']:.4f}")
    print(f"💾 Results saved to results/ directory")