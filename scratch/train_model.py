import fasttext
import os

def train():
    input_file = "train_data.txt"
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run scratch/export_for_fasttext.py first!")
        return

    print("🧠 Training FastText model...")
    # Train the model
    # epoch=25: number of times to see the data
    # lr=1.0: learning rate
    # wordNgrams=2: captures 2-word contexts (e.g., 'shopee food')
    model = fasttext.train_supervised(
        input=input_file, 
        epoch=25, 
        lr=1.0, 
        wordNgrams=2, 
        verbose=2, 
        minCount=1
    )

    print("💾 Saving base model (model.bin)...")
    model.save_model("model.bin")

    print("📉 Quantizing model for VPS efficiency...")
    # This compresses the model significantly (e.g., 100MB -> 10MB)
    # with almost zero loss in accuracy for binary classification.
    model.quantize(input=input_file, retrain=True)
    
    output_model = "model.ftz"
    model.save_model(output_model)

    print(f"\n✨ Success! '{output_model}' created.")
    print("Next step: Upload 'model.ftz' to your VPS and implement the 'Traffic Cop' logic.")

if __name__ == "__main__":
    train()
