#!/usr/bin/env python3
"""
CNN Retinal Test Image Page
Basic Model v4 - Diabetic Retinopathy Classification
"""

import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

class RetinalCNNTester:
    """Test page for retinal CNN model predictions"""
    
    def __init__(self, model_path="basic_modelv4_final.h5"):
        """Initialize the CNN tester"""
        self.model_path = model_path
        self.model = None
        self.class_labels = {
            0: "No DR", 
            1: "Mild DR", 
            2: "Moderate DR",
            3: "Severe DR", 
            4: "Proliferative DR"
        }
        self.confidence_threshold = 0.5
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained CNN model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                print(f"Model parameters: {self.model.count_params():,}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path, target_size=(512, 512)):
        """Preprocess image for CNN prediction"""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def predict_single_image(self, image_path):
        """Predict diabetic retinopathy for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_label': self.class_labels[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': predictions[0].tolist(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def predict_batch(self, image_directory, max_images=50):
        """Predict for multiple images in a directory"""
        image_dir = Path(image_directory)
        if not image_dir.exists():
            raise ValueError(f"Directory not found: {image_directory}")
        
        # Find image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(ext)))
        
        if not image_files:
            print(f"No image files found in {image_directory}")
            return []
        
        # Limit number of images
        image_files = image_files[:max_images]
        print(f"Processing {len(image_files)} images...")
        
        results = []
        for i, img_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
            result = self.predict_single_image(img_path)
            if result:
                results.append(result)
        
        return results
    
    def display_prediction(self, result):
        """Display prediction result with image"""
        if result is None:
            print("No result to display")
            return
        
        # Load and display image
        img = cv2.imread(str(result['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        # Image subplot
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Input Image: {Path(result['image_path']).name}")
        plt.axis('off')
        
        # Prediction bars
        plt.subplot(1, 2, 2)
        classes = list(self.class_labels.values())
        probabilities = result['all_probabilities']
        
        colors = ['green' if i == result['predicted_class'] else 'lightblue' 
                 for i in range(len(classes))]
        
        bars = plt.bar(classes, probabilities, color=colors)
        plt.title(f'Prediction: {result["predicted_label"]}\nConfidence: {result["confidence"]:.3f}')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print(f"\nDetailed Results:")
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Timestamp: {result['timestamp']}")
    
    def save_results(self, results, output_file="predictions.csv"):
        """Save prediction results to CSV"""
        if not results:
            print("No results to save")
            return
        
        # Convert to DataFrame
        df_data = []
        for result in results:
            row = {
                'image_path': result['image_path'],
                'image_name': Path(result['image_path']).name,
                'predicted_class': result['predicted_class'],
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                'timestamp': result['timestamp']
            }
            
            # Add individual class probabilities
            for i, prob in enumerate(result['all_probabilities']):
                row[f'prob_{self.class_labels[i]}'] = prob
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(df['predicted_label'].value_counts())
    
    def test_model_performance(self, test_images_dir, ground_truth_csv=None):
        """Test model performance if ground truth is available"""
        results = self.predict_batch(test_images_dir)
        
        if ground_truth_csv and os.path.exists(ground_truth_csv):
            # Load ground truth
            df_truth = pd.read_csv(ground_truth_csv)
            id_to_label = dict(zip(df_truth['id_code'], df_truth['diagnosis']))
            
            # Calculate accuracy
            correct = 0
            total = 0
            
            for result in results:
                image_name = Path(result['image_path']).stem
                if image_name in id_to_label:
                    true_label = int(id_to_label[image_name])
                    predicted_label = result['predicted_class']
                    
                    if true_label == predicted_label:
                        correct += 1
                    total += 1
            
            if total > 0:
                accuracy = correct / total
                print(f"\nModel Performance:")
                print(f"Correct predictions: {correct}/{total}")
                print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        return results


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="CNN Retinal Test Image Page")
    parser.add_argument("--model", default="basic_modelv4_final.h5", 
                       help="Path to trained model file")
    parser.add_argument("--image", help="Single image to predict")
    parser.add_argument("--directory", help="Directory of images to predict")
    parser.add_argument("--max_images", type=int, default=50, 
                       help="Maximum number of images to process")
    parser.add_argument("--output", default="predictions.csv", 
                       help="Output CSV file for results")
    parser.add_argument("--ground_truth", help="CSV file with ground truth labels")
    parser.add_argument("--show_plots", action="store_true", 
                       help="Display plots for predictions")
    
    args = parser.parse_args()
    
    # Initialize tester
    try:
        tester = RetinalCNNTester(model_path=args.model)
    except Exception as e:
        print(f"Failed to initialize tester: {e}")
        return
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
        
        print(f"Predicting single image: {args.image}")
        result = tester.predict_single_image(args.image)
        
        if result:
            if args.show_plots:
                tester.display_prediction(result)
            else:
                print(f"Prediction: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
    
    # Directory batch prediction
    elif args.directory:
        if not os.path.exists(args.directory):
            print(f"Directory not found: {args.directory}")
            return
        
        print(f"Predicting images in directory: {args.directory}")
        results = tester.predict_batch(args.directory, max_images=args.max_images)
        
        if results:
            tester.save_results(results, args.output)
            
            # Test performance if ground truth provided
            if args.ground_truth:
                tester.test_model_performance(args.directory, args.ground_truth)
    
    else:
        print("Please provide either --image or --directory argument")
        parser.print_help()


if __name__ == "__main__":
    # Interactive mode if no command line arguments
    if len(os.sys.argv) == 1:
        print("CNN Retinal Test Image Page - Interactive Mode")
        print("=" * 50)
        
        # Initialize tester
        model_path = input("Enter model path (or press Enter for 'basic_modelv4_final.h5'): ").strip()
        if not model_path:
            model_path = "basic_modelv4_final.h5"
        
        try:
            tester = RetinalCNNTester(model_path=model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            exit(1)
        
        while True:
            print("\nOptions:")
            print("1. Predict single image")
            print("2. Predict batch of images")
            print("3. Test model performance")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    result = tester.predict_single_image(image_path)
                    if result:
                        tester.display_prediction(result)
                else:
                    print("Image file not found")
            
            elif choice == "2":
                directory = input("Enter directory path: ").strip()
                max_images = input("Max images (default 50): ").strip()
                max_images = int(max_images) if max_images.isdigit() else 50
                
                if os.path.exists(directory):
                    results = tester.predict_batch(directory, max_images)
                    if results:
                        output_file = input("Output CSV filename (default 'predictions.csv'): ").strip()
                        if not output_file:
                            output_file = "predictions.csv"
                        tester.save_results(results, output_file)
                else:
                    print("Directory not found")
            
            elif choice == "3":
                directory = input("Enter test images directory: ").strip()
                csv_path = input("Enter ground truth CSV path: ").strip()
                
                if os.path.exists(directory):
                    results = tester.test_model_performance(directory, csv_path)
                else:
                    print("Directory not found")
            
            elif choice == "4":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice")
    
    else:
        main()