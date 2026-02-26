def log_strategy(strategy_name, base_accuracy, base_size, base_gzip, base_params, base_filters):
    print(f"\n{'='*100}")
    print(f"{strategy_name}:")
    print(f"{'='*100}")
    print(f"   Accuracy:          {base_accuracy*100:.2f}%")
    print(f"   Raw Size:          {base_size:.2f} KB")
    print(f"   Gzipped Size:      {base_gzip:.2f} KB")
    print(f"   Parameters:        {base_params:,}")
    print(f"   Conv Filters:      {base_filters}")