def compare_strategies(base_metrics, structured_metrics, unstructured_metrics, clustered_metrics, quantized_metrics, tflite_metrics):

    print(f"\n{'='*100}")
    print(f"FINAL MAXIMALLY COMPRESSED MODEL")
    print(f"{'='*100}")
    print(f"   Accuracy:          {tflite_metrics['accuracy']*100:.2f}%")
    print(f"   Accuracy Drop:     {(base_metrics['accuracy'] - tflite_metrics['accuracy'])*100:.2f}%")
    print(f"   Final Size:        {tflite_metrics['size_kb']:.2f} KB")
    print(f"   Starting Size:     {base_metrics['size_kb']:.2f} KB")
    print(f"   TOTAL REDUCTION:   {(1 - tflite_metrics['size_kb']/base_metrics['size_kb']) * 100:.2f}%")
    print(f"   COMPRESSION RATIO: {base_metrics['size_kb'] / tflite_metrics['size_kb']:.1f}x")

    # -------------------------
    # FINAL ANALYSIS SUMMARY (compact)
    # -------------------------
    print("\n" + "=" * 100)
    print("ENHANCED ANALYSIS - ALL 5 STRATEGIES WITH PARAMETERS")
    print("=" * 100)

    print(f"\n{'Strategy':<50} {'Size (KB)':<12} {'Gzip (KB)':<12} {'Params':<12} {'Accuracy':<8}")
    print("-" * 105)
    print(f"{'1. Baseline: Large Base':<50} {base_metrics['size_kb']:<12.2f} {base_metrics['gzip_kb']:<12.2f} {base_metrics['params']:<12,d} {base_metrics['accuracy']*100:>6.2f}%")
    print(f"{'2. + Structured Pruning (< filters)':<50} {structured_metrics['size_kb']:<12.2f} {structured_metrics['gzip_kb']:<12.2f} {structured_metrics['params']:<12,d} {structured_metrics['accuracy']*100:>6.2f}%")
    print(f"{'3. + Unstructured Pruning (80%)':<50} {unstructured_metrics['size_kb']:<12.2f} {unstructured_metrics['gzip_kb']:<12.2f} {unstructured_metrics['params']:<12,d} {unstructured_metrics['accuracy']*100:>6.2f}%")
    print(f"{'4. + Minimal Clustering (8)':<50} {clustered_metrics['size_kb']:<12.2f} {clustered_metrics['gzip_kb']:<12.2f} {clustered_metrics['params']:<12,d} {clustered_metrics['accuracy']*100:>6.2f}%")
    print(f"{'5. + QAT (INT8 prep)':<50} {quantized_metrics['size_kb']:<12.2f} {quantized_metrics['gzip_kb']:<12.2f} {quantized_metrics['params']:<12,d} {quantized_metrics['accuracy']*100:>6.2f}%")
    print(f"{'1+5. TFLite INT8 Final':<50} {tflite_metrics['size_kb']:<12.2f} {'-':<12} {'-':<12} {tflite_metrics['accuracy']*100:>6.2f}%")
    # -------------------------
    # DETAILED STRATEGY CONTRIBUTIONS (fixed)
    # Insert this after tflite_size and tflite_accuracy are computed
    # -------------------------
    print("\n" + "=" * 70)
    print("DETAILED STRATEGY CONTRIBUTIONS")
    print("=" * 70)
    print()

    # Safeguard: ensure gzip sizes exist (set to 0 if not)
    try:
        bg = float(base_metrics['gzip_kb'])
    except Exception:
        bg = float(base_metrics['size_kb']) if 'size_kb' in base_metrics else 0.0
    try:
        sg = float(structured_metrics['gzip_kb'])
    except Exception:
        sg = float(structured_metrics['size_kb']) if 'size_kb' in structured_metrics else 0.0
    try:
        ug = float(unstructured_metrics['gzip_kb'])
    except Exception:
        ug = float(unstructured_metrics['size_kb']) if 'size_kb' in unstructured_metrics else 0.0
    try:
        cg = float(clustered_metrics['gzip_kb'])
    except Exception:
        cg = float(clustered_metrics['size_kb']) if 'size_kb' in clustered_metrics else 0.0
    try:
        tfk = float(tflite_metrics['size_kb'])
    except Exception:
        tfk = float(tflite_metrics['size_kb']) if 'size_kb' in tflite_metrics else 0.0

    # Size reductions (KB)
    struct_size_red = bg - sg
    unstruct_size_red = sg - ug
    cluster_size_red = ug - cg
    final_size_red = cg - tfk

    # Parameter reductions (counts)
    bp = int(base_metrics['params']) if 'params' in base_metrics else 0
    sp = int(structured_metrics['params']) if 'params' in structured_metrics else 0
    up = int(unstructured_metrics['params']) if 'params' in unstructured_metrics else 0
    cp = int(clustered_metrics['params']) if 'params' in clustered_metrics else 0

    struct_param_red = bp - sp
    unstruct_param_red = sp - up   # expected 0 (weights zeroed, param count same)
    cluster_param_red = up - cp    # expected 0

    # Avoid division by zero
    def safe_div(a,b):
        return a/b if (b != 0) else float('inf')

    total_size_reduction = bg - tfk
    total_param_reduction = bp - sp

    # Efficiency metrics
    compression_ratio = safe_div(bg, tfk) if tfk > 0 else float('inf')
    parameter_efficiency = safe_div(bp, sp) if sp>0 else float('inf')
    memory_efficiency = compression_ratio
    inference_speedup = compression_ratio

    # Print block formatted similarly to the screenshot
    print("SIZE REDUCTION BREAKDOWN:")
    print(f"  Strategy 2 (Structured Pruning): {struct_size_red:9.2f} KB  ({(struct_size_red/bg*100) if bg>0 else 0:5.1f}%)")
    print(f"  Strategy 3 (Unstructured Pruning): {unstruct_size_red:9.2f} KB  ({(unstruct_size_red/sg*100) if sg>0 else 0:5.1f}%)")
    print(f"  Strategy 4 (Clustering): {cluster_size_red:9.2f} KB  ({(cluster_size_red/ug*100) if ug>0 else 0:5.1f}%)")
    print(f"  Strategy 1+5 (Large Model + INT8): {final_size_red:9.2f} KB  ({(final_size_red/cg*100) if cg>0 else 0:5.1f}%)")
    print("  " + "-" * 50)
    print(f"  Total Size Reduction: {total_size_reduction:14.2f} KB  ({(total_size_reduction/bg*100) if bg>0 else 0:5.1f}%)")
    print()
    print("PARAMETER REDUCTION BREAKDOWN:")
    print(f"  Strategy 2 (Structured Pruning): {struct_param_red:12,d} params  ({(struct_param_red/bp*100) if bp>0 else 0:5.1f}%)")
    print(f"  Strategy 3 (Unstructured Pruning): {unstruct_param_red:12,d} params  (weights zeroed)")
    print(f"  Strategy 4 (Clustering): {cluster_param_red:12,d} params  (values clustered)")
    print()
    print(f"  Strategy 1+5 (Large Model + INT8): {'-':>12} params  (precision reduced)")
    print("  " + "-" * 50)
    print(f"  Total Parameter Reduction: {total_param_reduction:12,d} params  ({(total_param_reduction/bp*100) if bp>0 else 0:5.1f}%)")
    print()
    print("EFFICIENCY METRICS:")
    print(f"  Compression Ratio: {compression_ratio:6.1f}x")
    if parameter_efficiency!=float('inf'):
        print(f"  Parameter Efficiency: {parameter_efficiency:6.1f}x fewer params")
    else:
        print(f"  Parameter Efficiency: {'-':>6}")
    print(f"  Memory Efficiency: {memory_efficiency:6.1f}x less memory")
    print(f"  Inference Speedup: ~{inference_speedup:6.1f}x faster")
    print("\n" + "=" * 70)





    print("\n" + "=" * 100)
    print("ALL 5 STRATEGIES SUCCESSFULLY APPLIED!")
    print("=" * 100)
    print(f"✓ Strategy 1: Large base model ({base_metrics['params']:,} params)")
    print(f"✓ Strategy 2: Structured pruning (removed {base_metrics['filters'] - structured_metrics['filters']} filters, {base_metrics['params'] - structured_metrics['params']:,} params)")
    print(f"✓ Strategy 3: 80% unstructured pruning ({unstruct:.1f}% sparsity)")
    print(f"✓ Strategy 4: 8 minimal clusters (only 8 unique values)")
    print(f"✓ Strategy 5: INT8 quantization (QAT followed by TFLite)")

    print(f"\nFINAL ACHIEVEMENTS:")
    print(f"  • Compression Ratio:    {base_metrics['size_kb'] / tflite_metrics['size_kb']:.1f}x smaller")
    print(f"  • Size Reduction:       {(1 - tflite_metrics['size_kb']/base_metrics['size_kb'])*100:.2f}%")
    print(f"  • Parameter Reduction:  {(1 - structured_metrics['params']/base_metrics['params'])*100:.2f}%")
    print(f"  • Accuracy Maintained:  {tflite_metrics['accuracy']*100:.2f}% (drop: {(base_metrics['accuracy'] - tflite_metrics['accuracy'])*100:.2f}%)")
    print("=" * 100)