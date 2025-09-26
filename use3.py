import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import argparse
import os

class ECGClassifier:

    CONFIDENCE_THRESHOLD = 0.8
    BATCH_PRINT_INTERVAL = 100
    MAX_TEXT_LENGTH = 128

    def __init__(self, model_path="./ecg_classifier_final"):
        print(f"Model loading from {model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            print(f"Model successfully loaded on: {self.device}")
            
        except Exception as e:
            print(f"Model loading error: {e}")
            raise
        
    def create_ecg_text(self, rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis):
        return f"ECG reading: RR interval {rr_interval}ms, P wave from {p_onset}ms to {p_end}ms, QRS complex from {qrs_onset}ms to {qrs_end}ms, T wave ends at {t_end}ms, axes P:{p_axis}° QRS:{qrs_axis}° T:{t_axis}°"
    

    def predict_single(self, rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis):
            text = self.create_ecg_text(rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis)
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.MAX_TEXT_LENGTH)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'label': 'Healthy' if predicted_class == 1 else 'Potentially anomalous',
                'probability_healthy': probabilities[0][1].item(),
                'probability_abnormal': probabilities[0][0].item()
            }

    def predict_from_csv(self, csv_path, output_path=None):
        #Prediction for CSV file
        print(f"Reading data from {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} sample(s)")
        except Exception as e:
            print(f"CSV file reading error: {e}")
            return None
        
        # Columns exist checking
        required_cols = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Columns doesn't exist: {missing_cols}")
            return None
        
        results = []
        
        print("Processing sample(s)...")
        for idx, row in df.iterrows():
            try:
                result = self.predict_single(
                    row['rr_interval'], row['p_onset'], row['p_end'],
                    row['qrs_onset'], row['qrs_end'], row['t_end'],
                    row['p_axis'], row['qrs_axis'], row['t_axis']
                )
                result['sample_id'] = idx
                results.append(result)
                
                if (idx + 1) % self.BATCH_PRINT_INTERVAL == 0:
                    print(f"Processed {idx + 1} sample(s)...")
                    
            except Exception as e:
                print(f"There is an error in sample {idx}: {e}")
                results.append({
                    'sample_id': idx,
                    'prediction': None,
                    'label': 'Error',
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        # Results saving
        if output_path is None:
            output_path = csv_path.replace('.csv', '_predictions.csv')
        
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        self.print_statistics(results_df)
        
        return results_df
    
    def print_statistics(self, results_df):
        print("\n" + "="*20)
        print("PREDICTION STATISTICS")
        print("="*20)
        
        valid_results = results_df.dropna(subset=['prediction'])
        
        if len(valid_results) == 0:
            print("There is no valid predictions!")
            return
        
        total = len(valid_results)
        healthy = (valid_results['prediction'] == 1).sum()
        abnormal = (valid_results['prediction'] == 0).sum()
        
        print(f"Total: {total}")
        print(f"|- Healthy: {healthy} ({healthy/total*100:.1f}%)")
        print(f"└─ Potentially anomalous: {abnormal} ({abnormal/total*100:.1f}%)")
        
        if 'confidence' in valid_results.columns:
            avg_conf = valid_results['confidence'].mean()
            print(f"Average confidence: {avg_conf:.3f}")
            
            high_conf = (valid_results['confidence'] > self.CONFIDENCE_THRESHOLD).sum()
            print(f"Maximal confidence (>0.8): {high_conf} ({high_conf/total*100:.1f}%)")

def main():
    print("\n=== ECG Classifier ===")
    
    parser = argparse.ArgumentParser(description='ECG Classifier')
    parser.add_argument('input_file',
			nargs='?',
			default='ecg_test_data.csv',
			help='Path to .csv file with ECG data (default: ecg_data.csv)')

    parser.add_argument('-o','--output',
			default=None,
			help='Path for saving results (default: auto-generated from input file)')
    parser.add_argument('-m','--model',
			default='./ecg_classifier_final',
			help='Path to model (default: ./ecg_classifier_final)')
    args = parser.parse_args()

    '''model_path = "./ecg_classifier_final"'''
    
    try:
        classifier = ECGClassifier(args.model)
    except Exception as e:
        print(f"Model loading isn't successful: {e}")
        return 1
    
    print(f"\nCSV file processing: {args.input_file}")
    
    if not os.path.exists(args.input_file):
        print(f"File {args.input_file} doesn't exist!")
        
        # If default file was entered, creating test data
        if args.input_file == 'ecg_test_data.csv':
            print("Creating test file...")
            test_data = pd.DataFrame({
                'rr_interval': [850, 920, 780, 950, 800],
                'p_onset': [40, 45, 35, 50, 38],
                'p_end': [160, 170, 150, 180, 155],
                'qrs_onset': [190, 200, 180, 210, 185],
                'qrs_end': [280, 290, 270, 300, 275],
                't_end': [590, 610, 570, 630, 580],
                'p_axis': [60, 65, 55, 70, 58],
                'qrs_axis': [40, 45, 35, 50, 38],
                't_axis': [20, 25, 15, 30, 18]
            })
            test_data.to_csv(args.input_file, index=False)
            print(f"Test file {args.input_file} successfully created.")
        else:
            return 1
    
    # File processing
    results = classifier.predict_from_csv(args.input_file, args.output)
    
    if results is not None:
        print("\nFirst results:")
        print(results[['sample_id', 'label', 'confidence']].head(5))
        return 0
    else:
        print("Processing error")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
