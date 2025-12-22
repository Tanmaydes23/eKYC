# Quick Test Script for Complete eKYC System
# Run this in Google Colab after running complete_ekyc_system.py

print("🧪 Testing Complete eKYC System\n")

# Example 1: Test with sample images from your dataset
print("="*80)
print("Test 1: Verify Real ID vs Real Selfie (Should PASS)")
print("="*80)

# Assuming you have sample images in your dataset
genuine_image_1 = '/content/drive/MyDrive/Kaggle/Duplication_Dataset/sample1.jpg'  # Replace with actual path
genuine_image_2 = '/content/drive/MyDrive/Kaggle/Duplication_Dataset/sample2.jpg'  # Replace with actual path

results1 = ekyc_system.complete_verification(
    id_image_path=genuine_image_1,
    selfie_image_path=genuine_image_2,
    deepfake_threshold=0.4,  # Lower threshold for 2-epoch model
    identity_threshold=0.5,
    liveness_threshold=0.2
)

print(f"\n📊 Final Result: {'✅ PASS' if results1['final_decision']['pass'] else '❌ FAIL'}")
print(f"Confidence: {results1['final_decision']['confidence']:.2%}")

# =============================================================================
print("\n\n" + "="*80)
print("Test 2: Verify Fake Image (Should FAIL deepfake check)")
print("="*80)

fake_image = '/content/drive/MyDrive/Kaggle/Forgery_Dataset/sample1.jpg'  # Replace with actual path

results2 = ekyc_system.complete_verification(
    id_image_path=genuine_image_1,
    selfie_image_path=fake_image,
    deepfake_threshold=0.4,
    identity_threshold=0.5,
    liveness_threshold=0.2
)

print(f"\n📊 Final Result: {'✅ PASS' if results2['final_decision']['pass'] else '❌ FAIL'}")
print(f"Confidence: {results2['final_decision']['confidence']:.2%}")

# =============================================================================
print("\n\n" + "="*80)
print("📋 Individual Component Scores:")
print("="*80)

print("\n1️⃣ Deepfake Detection Module:")
test_image = genuine_image_1
score = ekyc_system.detect_deepfake(test_image)
print(f"   Authenticity Score: {score:.2%} (higher = more real)")

print("\n2️⃣ Identity Verification Module:")
match_score = ekyc_system.verify_identity(genuine_image_1, genuine_image_2)
print(f"   Match Score: {match_score:.2%} (higher = same person)")

print("\n3️⃣ Liveness Detection Module:")
liveness = ekyc_system.detect_liveness(genuine_image_1)
print(f"   Liveness Score: {liveness:.2%} (higher = more likely live)")

print("\n\n" + "="*80)
print("✅ Testing Complete!")
print("="*80)
print("\n💡 Next Steps:")
print("   1. Test with your own images")
print("   2. Adjust thresholds based on results")
print("   3. Train more epochs for better deepfake accuracy")
print("   4. Create Streamlit demo app")
