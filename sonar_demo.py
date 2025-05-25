import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
import random

class SonarThreatDetector:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.threat_log = []
    
    def train_system(self, filepath):
        """Train the sonar threat detection system"""
        print("🔄 INITIALIZING SONAR THREAT DETECTION SYSTEM...")
        print("=" * 60)
        
        # Load historical sonar data
        df = pd.read_csv(filepath)
        print(f"📊 Loading historical sonar database: {df.shape[0]} samples")
        
        # Prepare data
        X = df.drop('R', axis=1)
        y = df['R'].map({'R': 0, 'M': 1})  # 0=Rock, 1=Mine
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("🧠 Training AI classification model...")
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ System trained successfully!")
        print(f"🎯 Detection accuracy: {accuracy:.1%}")
        print(f"📈 Training samples: {len(X_train)}")
        print(f"🔍 Test samples: {len(X_test)}")
        
        self.is_trained = True
        return accuracy
    
    def scan_contact(self, sonar_data, contact_id=None):
        """Analyze a single sonar contact"""
        if not self.is_trained:
            return "❌ ERROR: System not trained"
        
        # Get prediction and confidence
        prediction = self.model.predict(sonar_data.reshape(1, -1))[0]
        confidence = self.model.predict_proba(sonar_data.reshape(1, -1))[0]
        
        # Determine threat level
        threat_type = "MINE" if prediction == 1 else "ROCK"
        confidence_score = confidence[prediction]
        
        # Create threat assessment
        if threat_type == "MINE":
            if confidence_score > 0.8:
                alert_level = "🚨 HIGH THREAT"
                action = "IMMEDIATE EVASIVE ACTION"
            elif confidence_score > 0.6:
                alert_level = "⚠️  MODERATE THREAT"
                action = "PROCEED WITH CAUTION"
            else:
                alert_level = "⚡ LOW THREAT"
                action = "MONITOR CLOSELY"
        else:
            alert_level = "✅ NO THREAT"
            action = "SAFE TO PROCEED"
        
        # Log the contact
        contact_info = {
            'id': contact_id or f"CONTACT-{len(self.threat_log)+1:03d}",
            'type': threat_type,
            'confidence': confidence_score,
            'alert': alert_level,
            'action': action,
            'timestamp': time.strftime("%H:%M:%S")
        }
        
        self.threat_log.append(contact_info)
        return contact_info
    
    def simulate_patrol(self, test_data, num_contacts=10):
        """Simulate a submarine patrol with multiple sonar contacts"""
        print("\n🚢 BEGINNING PATROL SIMULATION")
        print("=" * 60)
        
        # Select random contacts from test data
        indices = random.sample(range(len(test_data)), min(num_contacts, len(test_data)))
        
        for i, idx in enumerate(indices, 1):
            time.sleep(0.5)  # Simulate time between contacts
            
            sonar_reading = test_data.iloc[idx].drop('R').values
            actual_type = "MINE" if test_data.iloc[idx]['R'] == 'M' else "ROCK"
            
            print(f"\n📡 SONAR CONTACT #{i:02d} DETECTED")
            print("-" * 40)
            
            result = self.scan_contact(sonar_reading, f"PATROL-{i:02d}")
            
            print(f"Contact ID: {result['id']}")
            print(f"Time: {result['timestamp']}")
            print(f"Classification: {result['type']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Alert Level: {result['alert']}")
            print(f"Recommended Action: {result['action']}")
            print(f"Actual Type: {actual_type} {'✅' if result['type'] == actual_type else '❌'}")
    
    def generate_mission_report(self):
        """Generate a mission summary report"""
        if not self.threat_log:
            return "No contacts logged"
        
        print("\n📋 MISSION SUMMARY REPORT")
        print("=" * 60)
        
        total_contacts = len(self.threat_log)
        mines_detected = sum(1 for log in self.threat_log if log['type'] == 'MINE')
        rocks_detected = total_contacts - mines_detected
        high_threats = sum(1 for log in self.threat_log if 'HIGH' in log['alert'])
        
        print(f"Total Contacts Analyzed: {total_contacts}")
        print(f"Potential Mines Detected: {mines_detected}")
        print(f"Rocks Identified: {rocks_detected}")
        print(f"High-Priority Threats: {high_threats}")
        
        if high_threats > 0:
            print(f"\n⚠️  WARNING: {high_threats} high-priority threats detected!")
            print("Recommend immediate area evacuation and mine sweeping operations.")
        else:
            print("\n✅ Area appears relatively safe for navigation.")

def main():
    # Initialize the system
    detector = SonarThreatDetector()
    
    # Train the system
    accuracy = detector.train_system("data/sonar.csv")
    
    # Load test data
    test_data = pd.read_csv("data/sonar.csv")
    
    # Run patrol simulation
    detector.simulate_patrol(test_data, num_contacts=8)
    
    # Generate report
    detector.generate_mission_report()
    
    print(f"\n🔚 MISSION COMPLETE - System Accuracy: {accuracy:.1%}")

if __name__ == "__main__":
    main()