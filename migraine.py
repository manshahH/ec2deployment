import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import pickle # Import the pickle module
import os  

# Load dataset
df = pd.read_csv("migraine_symptom_classification.csv")

# Display basic info
print("\n🔍 Dataset Information:")
print(df.info())

# Check for missing values
print("\n❌ Missing Values per Column:")
print(df.isnull().sum())

# Convert categorical 'Type' column to numeric values
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])  # Assign unique numeric values to each migraine type

# Fill missing values with column mean (if needed)
df.fillna(df.mean(), inplace=True)

# Summary statistics
print("\n📊 Summary Statistics:")
print(df.describe())

# Visualize migraine type distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Type", data=df, palette="coolwarm")
plt.title("🧠 Distribution of Migraine Types")
plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)  # Map numeric labels back to migraine types
plt.show()

# Pairplot to visualize feature relationships
sns.pairplot(df, hue="Type", diag_kind="kde", palette="coolwarm")
plt.suptitle("🔎 Pairwise Relationships Between Features", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title("🔥 Feature Correlation Heatmap")
plt.show()

# Load dataset
# Encode target variable (categorical to numeric)
print("\n--- Preparing Data for Model Training ---")

# --- UPDATED: Explicitly define the 23 feature columns from your CSV ---
selected_feature_columns = [
    'Age', 'Duration', 'Frequency', 'Location', 'Character', 'Intensity',
    'Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory',
    'Dysphasia', 'Dysarthria', 'Vertigo', 'Tinnitus', 'Hypoacusis', 'Diplopia',
    'Defect', 'Ataxia', 'Conscience', 'Paresthesia', 'DPF'
]

# Separate features (X) and target (y)
X = df[selected_feature_columns] # Selects only the 23 specified features
y = df["Type"]

print(f"Number of features selected for X: {X.shape[1]}") # Should print 23

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")


# Normalize features
# This StandardScaler will be fitted on the 23 features and saved for deployment
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Fit on training data
X_test = scaler.transform(X_test)       # Transform test data using the fitted scaler
print("Features normalized using StandardScaler.")


# --- Save Preprocessing Objects (for deployment) ---
print("\n--- Saving Preprocessing Objects ---")
PREPROCESSING_DIR = "preprocessing_objects"
os.makedirs(PREPROCESSING_DIR, exist_ok=True)
print(f"Preprocessing objects will be saved in: {PREPROCESSING_DIR}")

# Save the fitted StandardScaler
scaler_filename = os.path.join(PREPROCESSING_DIR, "scaler.pkl")
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"StandardScaler saved to {scaler_filename}")

# Save the fitted LabelEncoder
label_encoder_filename = os.path.join(PREPROCESSING_DIR, "label_encoder.pkl")
with open(label_encoder_filename, 'wb') as file:
    pickle.dump(le, file)
print(f"LabelEncoder saved to {label_encoder_filename}")


# --- Model Definition ---
print("\n--- Defining Models ---")
models = {
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "params": {"n_estimators": 100, "random_state": 42}
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"max_iter": 1000}
    },
}

# --- Model Saving Directory for .pkl files ---
PKL_MODELS_DIR = "trained_models_pkl"
os.makedirs(PKL_MODELS_DIR, exist_ok=True)
print(f"Trained .pkl models will be saved in: {PKL_MODELS_DIR}")


# --- Train and Evaluate Each Model with MLflow Tracking ---
print("\n--- Training and Evaluating Models ---")
for name, model_info in models.items():
    model = model_info["model"]
    params = model_info["params"]

    # Start an MLflow run for each model
    with mlflow.start_run(run_name=name):
        print(f"\n--- Training {name} ---")

        # Log parameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state_split", 42) # For train_test_split
        mlflow.log_param("num_features", X.shape[1]) # Log the number of features used

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        print(f"{name} Accuracy: {accuracy:.2f}")

        # Generate and log classification report as an MLflow artifact
        report_str = "Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0)
        print(report_str)

        # Save classification report temporarily and log it
        report_filename = f"{name.lower().replace(' ', '_')}_classification_report.txt"
        with open(report_filename, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_filename)
        os.remove(report_filename) # Clean up the local file after logging

        # Generate and Log Confusion Matrix Plot as an MLflow artifact
        cm = confusion_matrix(y_test, y_pred)
        # Use le.classes_ to get actual migraine type names for display labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Confusion Matrix for {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Save confusion matrix plot temporarily and log it
        cm_plot_filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(cm_plot_filename)
        plt.close(fig) # Close the figure to free up memory
        mlflow.log_artifact(cm_plot_filename)
        os.remove(cm_plot_filename) # Clean up local file

        # Log the trained model to MLflow (for comprehensive tracking)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # Path within the MLflow run's artifact URI
            registered_model_name=f"{name}_MigraineClassifier" # Optional: Register model for versioning
        )

        # --- Save the model as a .pkl file (for direct deployment/loading) ---
        pkl_filename = os.path.join(PKL_MODELS_DIR, f"{name.lower().replace(' ', '_')}_model.pkl")
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {pkl_filename}")
        mlflow.log_artifact(pkl_filename) # Also log the .pkl file as an artifact in MLflow

        print(f"MLflow Run ID for {name}: {mlflow.active_run().info.run_id}")

print("\n--- All models trained, preprocessing objects, and artifacts logged ---")
print("Run 'mlflow ui' in your terminal to view the results.")
print("Now, you MUST update your main.py and app.py to reflect these 23 features.")
