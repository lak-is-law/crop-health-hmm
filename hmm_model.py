import numpy as np
from scipy.special import logsumexp

class GaussianHMM:
    """
    Gaussian Hidden Markov Model for sequential crop-health state detection.
    
    States:
        0: Healthy
        1: Stressed (Drought, Heat, Cold, Moisture Deficit)
        2: Disease-Suspected (Fungal Blight, Rust, Canopy Lesions)
    """
    
    STATE_NAMES = ["Healthy", "Stressed", "Disease-Suspected"]
    STATE_COLORS = ["#22c55e", "#eab308", "#ef4444"]

    def __init__(self, n_states=3, feature_names=None):
        self.n_states = n_states
        self.feature_names = feature_names or ["ndvi", "temperature_c", "humidity_pct", "soil_moisture_pct"]
        self.n_features = len(self.feature_names)
        
        # Initial state probabilities pi (predominantly Healthy at season start)
        self.pi = np.array([0.70, 0.25, 0.05], dtype=np.float64)
        
        # Transition probability matrix A [from_state, to_state]
        # High persistence on the diagonal with realistic agricultural state shifts
        self.A = np.array([
            [0.78, 0.17, 0.05],  # Healthy -> [Healthy, Stressed, Disease]
            [0.22, 0.65, 0.13],  # Stressed -> [Healthy, Stressed, Disease]
            [0.08, 0.22, 0.70]   # Disease  -> [Healthy, Stressed, Disease]
        ], dtype=np.float64)
        
        # Emission distributions: Multivariate Gaussian (means mu and diagonal covariance vars)
        # Features: [ndvi, temperature_c, humidity_pct, soil_moisture_pct]
        self.means = np.array([
            # State 0: Healthy (high NDVI, temperate weather, optimal humidity & moisture)
            [0.68, 23.5, 66.0, 48.0],
            # State 1: Stressed (depressed NDVI, elevated temp / drought or cold, dry soil)
            [0.46, 31.5, 45.0, 22.0],
            # State 2: Disease-Suspected (low/falling NDVI, high humidity, warm damp soil)
            [0.40, 26.5, 88.0, 60.0]
        ], dtype=np.float64)
        
        self.vars = np.array([
            # Variance per feature for state 0
            [0.015, 12.0, 50.0, 55.0],
            # Variance per feature for state 1
            [0.012, 18.0, 60.0, 40.0],
            # Variance per feature for state 2
            [0.014,  8.0, 35.0, 45.0]
        ], dtype=np.float64)

    def _log_gaussian_pdf(self, X, mean, var):
        """
        Computes log N(X | mean, diag(var)).
        X: shape (T, D)
        mean: shape (D,)
        var: shape (D,)
        Returns: array of shape (T,)
        """
        d = len(mean)
        var = np.maximum(var, 1e-4)
        log_norm = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)))
        diff = X - mean
        quad = -0.5 * np.sum((diff ** 2) / var, axis=1)
        return log_norm + quad

    def compute_log_emissions(self, X):
        """
        Computes log emission probabilities log P(O_t | S_t = k)
        X: shape (T, D)
        Returns: log_B of shape (T, n_states)
        """
        T = X.shape[0]
        log_B = np.zeros((T, self.n_states), dtype=np.float64)
        for k in range(self.n_states):
            log_B[:, k] = self._log_gaussian_pdf(X, self.means[k], self.vars[k])
        return log_B

    def forward(self, X):
        """
        Computes log forward probabilities alpha_t(i) = log P(O_1:t, S_t = i)
        Returns: log_alpha shape (T, n_states), total log-likelihood
        """
        T = X.shape[0]
        log_B = self.compute_log_emissions(X)
        log_pi = np.log(np.maximum(self.pi, 1e-12))
        log_A = np.log(np.maximum(self.A, 1e-12))
        
        log_alpha = np.zeros((T, self.n_states), dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]
        
        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_A[:, j]) + log_B[t, j]
                
        total_log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, total_log_likelihood

    def backward(self, X):
        """
        Computes log backward probabilities beta_t(i) = log P(O_t+1:T | S_t = i)
        Returns: log_beta shape (T, n_states)
        """
        T = X.shape[0]
        log_B = self.compute_log_emissions(X)
        log_A = np.log(np.maximum(self.A, 1e-12))
        
        log_beta = np.zeros((T, self.n_states), dtype=np.float64)
        log_beta[-1] = 0.0  # beta_T(i) = 1 => log_beta_T(i) = 0
        
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(log_A[i, :] + log_B[t+1, :] + log_beta[t+1, :])
                
        return log_beta

    def posterior_probabilities(self, X):
        """
        Forward-Backward Algorithm:
        Computes gamma_t(i) = P(S_t = i | O_1:T).
        Returns: gamma shape (T, n_states)
        """
        log_alpha, _ = self.forward(X)
        log_beta = self.backward(X)
        
        log_gamma = log_alpha + log_beta
        for t in range(X.shape[0]):
            log_gamma[t] -= logsumexp(log_gamma[t])
            
        gamma = np.exp(log_gamma)
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def viterbi(self, X):
        """
        Viterbi Algorithm:
        Computes the most likely sequence of hidden states q*_1:T = argmax P(q_1:T | O_1:T).
        Returns: best_path (list of int state indices), max_log_prob (float)
        """
        T = X.shape[0]
        log_B = self.compute_log_emissions(X)
        log_pi = np.log(np.maximum(self.pi, 1e-12))
        log_A = np.log(np.maximum(self.A, 1e-12))
        
        viterbi_vars = np.zeros((T, self.n_states), dtype=np.float64)
        backpointers = np.zeros((T, self.n_states), dtype=np.int32)
        
        viterbi_vars[0] = log_pi + log_B[0]
        
        for t in range(1, T):
            for j in range(self.n_states):
                candidates = viterbi_vars[t-1] + log_A[:, j]
                best_prev_state = np.argmax(candidates)
                backpointers[t, j] = best_prev_state
                viterbi_vars[t, j] = candidates[best_prev_state] + log_B[t, j]
                
        best_last_state = np.argmax(viterbi_vars[-1])
        max_log_prob = viterbi_vars[-1, best_last_state]
        
        best_path = [best_last_state]
        for t in range(T - 1, 0, -1):
            prev_state = backpointers[t, best_path[-1]]
            best_path.append(prev_state)
            
        best_path.reverse()
        return best_path, float(max_log_prob)

    def fit_supervised(self, sequences_X, sequences_states):
        """
        Fits HMM parameters directly using empirical frequencies from labeled sequences.
        """
        initial_counts = np.zeros(self.n_states)
        for states in sequences_states:
            initial_counts[states[0]] += 1
        self.pi = (initial_counts + 1e-2) / np.sum(initial_counts + 1e-2)
        
        trans_counts = np.zeros((self.n_states, self.n_states))
        for states in sequences_states:
            for t in range(len(states) - 1):
                trans_counts[states[t], states[t+1]] += 1
        for i in range(self.n_states):
            row_sum = np.sum(trans_counts[i])
            if row_sum > 0:
                self.A[i] = (trans_counts[i] + 1e-2) / (row_sum + self.n_states * 1e-2)
                
        for k in range(self.n_states):
            k_obs = []
            for X, states in zip(sequences_X, sequences_states):
                mask = (states == k)
                if np.any(mask):
                    k_obs.append(X[mask])
            if len(k_obs) > 0:
                k_data = np.vstack(k_obs)
                self.means[k] = np.mean(k_data, axis=0)
                self.vars[k] = np.maximum(np.var(k_data, axis=0), 1e-3)

    def fit_em(self, sequences_X, n_iter=15, tol=1e-4):
        """
        Baum-Welch Expectation-Maximization algorithm for unsupervised refinement.
        """
        log_A = np.log(np.maximum(self.A, 1e-12))
        prev_ll = -np.inf
        
        for iteration in range(n_iter):
            new_pi = np.zeros(self.n_states)
            new_A = np.zeros((self.n_states, self.n_states))
            gamma_sum = np.zeros(self.n_states)
            obs_weighted_sum = np.zeros((self.n_states, self.n_features))
            total_ll = 0.0
            
            for X in sequences_X:
                T = X.shape[0]
                log_alpha, ll = self.forward(X)
                log_beta = self.backward(X)
                log_B = self.compute_log_emissions(X)
                total_ll += ll
                
                log_gamma = log_alpha + log_beta
                for t in range(T):
                    log_gamma[t] -= logsumexp(log_gamma[t])
                gamma = np.exp(log_gamma)
                
                new_pi += gamma[0]
                
                for t in range(T - 1):
                    log_xi_t = (
                        log_alpha[t][:, None]
                        + log_A
                        + log_B[t+1][None, :]
                        + log_beta[t+1][None, :]
                    )
                    log_xi_t -= logsumexp(log_xi_t)
                    new_A += np.exp(log_xi_t)
                    
                for k in range(self.n_states):
                    gamma_k = gamma[:, k]
                    g_sum = np.sum(gamma_k)
                    gamma_sum[k] += g_sum
                    obs_weighted_sum[k] += np.sum(gamma_k[:, None] * X, axis=0)
                    
            self.pi = np.maximum(new_pi / np.sum(new_pi), 1e-4)
            self.pi /= np.sum(self.pi)
            
            for i in range(self.n_states):
                row_sum = np.sum(new_A[i])
                if row_sum > 0:
                    self.A[i] = np.maximum(new_A[i] / row_sum, 1e-4)
                    self.A[i] /= np.sum(self.A[i])
                    
            for k in range(self.n_states):
                if gamma_sum[k] > 1e-5:
                    self.means[k] = obs_weighted_sum[k] / gamma_sum[k]
                    diff_sq_sum = np.zeros(self.n_features)
                    for X in sequences_X:
                        log_alpha, _ = self.forward(X)
                        log_beta = self.backward(X)
                        log_gamma = log_alpha + log_beta
                        for t in range(X.shape[0]):
                            log_gamma[t] -= logsumexp(log_gamma[t])
                        gamma_k = np.exp(log_gamma)[:, k]
                        diff = X - self.means[k]
                        diff_sq_sum += np.sum(gamma_k[:, None] * (diff ** 2), axis=0)
                    self.vars[k] = np.maximum(diff_sq_sum / gamma_sum[k], 1e-3)
                    
            if abs(total_ll - prev_ll) < tol:
                break
            prev_ll = total_ll

    def predict_sequence(self, X):
        """Convenience method returning decoded state names and posteriors."""
        best_path_indices, log_prob = self.viterbi(X)
        posteriors = self.posterior_probabilities(X)
        state_names = [self.STATE_NAMES[idx] for idx in best_path_indices]
        return {
            "decoded_indices": best_path_indices,
            "decoded_states": state_names,
            "posteriors": posteriors,
            "log_likelihood": log_prob
        }
