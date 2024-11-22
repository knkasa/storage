def calculate_shapley_values_sampling(model, instance, baseline, n_samples=100):
    n_features = instance.shape[1]
    shapley_values = np.zeros(n_features)
    for i in range(n_features):
        shapley_value = 0.0
        for _ in range(n_samples):
            subset = np.random.choice([j for j in range(n_features) if j != i],
                                       size=np.random.randint(0, n_features), replace=False)
            diff = 0.0
            for base_row in baseline:
                mask_with_feature = base_row.copy()
                mask_without_feature = base_row.copy()
                for j in subset:
                    mask_with_feature[j] = instance[0, j]
                    mask_without_feature[j] = instance[0, j]
                mask_with_feature[i] = instance[0, i]
                diff += (model.predict(mask_with_feature[np.newaxis, :]) - 
                         model.predict(mask_without_feature[np.newaxis, :])) / baseline.shape[0]
            weight = (np.math.factorial(len(subset)) *
                      np.math.factorial(n_features - len(subset) - 1) /
                      np.math.factorial(n_features))
            shapley_value += weight * diff
        shapley_values[i] = shapley_value / n_samples
    return shapley_values
