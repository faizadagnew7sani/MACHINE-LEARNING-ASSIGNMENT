# STEP 6: Make predictions

y_pred = model.predict(X_test)


# STEP 7: Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("\nAccuracy:")
print(f"{accuracy:.4f}")


# STEP 8: Calculate Fault Precision and Fault Recall

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    y_pred,
    labels=["Fault"],
    average=None,
    zero_division=0
)

fault_precision = precision[0]
fault_recall = recall[0]

print("\nFault Precision:")
print(f"{fault_precision:.4f}")

print("\nFault Recall:")
print(f"{fault_recall:.4f}")


# STEP 9: Calculate Macro F1-score

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\nMacro F1-score:")
print(f"{macro_f1:.4f}")


# STEP 10: Classification Report

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Normal", "Warning", "Fault"],
        zero_division=0
    )
)
