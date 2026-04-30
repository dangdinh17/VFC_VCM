
\section{Classical Regularization Methods}
We now outline key classical methods, many of which have analogues in deep learning.

\subsection{Tikhonov (L2) Regularization}
Tikhonov regularization~\cite{Tikhonov1963} solves:
\[
    \min_{f\in\mathcal{H}} \;\|A f - g\|^2 + \lambda \|f\|^2.
\]
The unique solution is 
\[
    f_\lambda = (A^T A + \lambda I)^{-1} A^T g.
\]
This smooths the solution by penalizing its norm. In Bayesian terms, assuming $f\sim\mathcal{N}(0,\sigma_f^2 I)$ and $w\sim\mathcal{N}(0,\sigma_w^2 I)$ leads to this formulation with $\lambda=\sigma_w^2/\sigma_f^2$. As $\lambda\to 0$ (with noise $\to0$), $f_\lambda$ approaches the minimum-norm solution $A^+g$. Choosing $\lambda$ is critical: too large oversmooths, too small overfits. Common selection rules include cross-validation, L-curve, or the discrepancy principle.

\subsection{Truncated SVD (Spectral Cut-off)}
Via the SVD $A=U\Sigma V^T$, one can write the least-squares solution $A^+g = V\Sigma^{-1}U^T g$. Small singular values $\sigma_i$ cause instability. Truncated SVD (TSVD) addresses this by omitting terms with $\sigma_i<\sigma_{\min}$ or keeping only top $k$ components:
\[
    f_k = \sum_{i=1}^k \frac{u_i^T g}{\sigma_i} v_i.
\]
This is equivalent to projection onto $\mathrm{span}\{v_1,\dots,v_k\}$. P.\ C.\ Hansen showed that TSVD performs well under a variety of noise models. Its advantage is a clear cutoff of ill-conditioned modes; its drawback is the need to choose $k$ (analogous to model order) and the computational cost of SVD.

\subsection{Iterative Regularization (Landweber)}
The Landweber iteration was originally proposed in 1951. It is simply gradient descent on $\|Af - g\|^2$. With a suitable step-size $\omega$, the iterations converge to a regularized solution if stopped early. Pseudocode is given in Algorithm~\ref{alg:landweber}. A key property is semi-convergence: the iterate error first decreases then increases as noise is fitted. Thus, one must stop when the residual matches the noise level. The stopping index $k$ plays the role of $\lambda$ in controlling stability.

\begin{algorithm}[h]
\caption{Landweber Iteration for $A f = g$}
\label{alg:landweber}
\begin{algorithmic}
\REQUIRE Linear operator $A$, data $g$, step $\omega$ (e.g.\ $<2/\|A\|^2$), tolerance $\epsilon$.
\STATE Initialize $f^{(0)} \gets 0$.
\FOR{$k=0,1,2,\ldots$}
    \STATE $r^{(k)} \gets A f^{(k)} - g$  \COMMENT{Residual}
    \STATE $f^{(k+1)} \gets f^{(k)} - \omega\,A^T r^{(k)}$
    \IF{$\|r^{(k)}\| < \epsilon$ (noise level) }
        \STATE break  \COMMENT{Stop early to regularize}
    \ENDIF
\ENDFOR
\RETURN $f^{(k+1)}$.
\end{algorithmic}
\end{algorithm}

\subsection{Bayesian and Variational Methods}
In a broader sense, one may choose other priors. For example, a Laplace prior on $f$ leads to $\ell_1$ regularization, promoting sparsity in a given basis (e.g.\ wavelets~\cite{Donoho1995}). Total Variation (TV)~\cite{Rudin1992} regularization uses an $\ell_1$ penalty on image gradients:
\[
    \min_f \; \|Af - g\|^2 + \alpha \|\nabla f\|_1,
\]
and is celebrated for edge-preserving denoising. In Bayesian terms, TV corresponds to a prior favoring piecewise-constant images. Many classical methods (spline smoothing, Gaussian Markov random fields, etc.) fit into this variational framework. These handcrafted priors are powerful when well-chosen, but may fail to capture complex natural image statistics.

\section{Mapping to Modern Deep Learning}
Deep learning for inverse problems often uses neural networks trained end-to-end or as learned iterative schemes. Regularization in this context appears in several guises. We outline the main ones and their classical analogues.

\subsection{Weight Decay (L2 Regularization)}
Weight decay~\cite{Krogh1992} adds an $\ell_2$ penalty on network weights $W$ during training, modifying the loss to $L_{\rm data}(W) + \frac{\alpha}{2}\|W\|^2$. This is exactly the Tikhonov penalty on parameters. In linear networks, Krogh and Hertz proved that weight decay suppresses unimportant directions and reduces sensitivity to label noise. In effect, weight decay biases the solution towards the smallest-norm parameters consistent with the data, just as classical ridge regression selects minimum-norm solutions. Thus weight decay in deep nets inherits the smoothing effect of classical L2 regularization, mitigating overfitting. The hyperparameter $\alpha$ must be tuned (often by validation).

\subsection{Dropout and Noise Injection}
Dropout~\cite{Srivastava2014} randomly sets a fraction $p$ of hidden units (or weights) to zero during each training pass. Equivalently, it injects multiplicative Bernoulli noise into activations. The result is an ensemble effect: at test time, one approximately averages over many thinned networks. Practically, dropout discourages complex co-adaptations of neurons. Intuitively, dropout is akin to adding noise to the model, which can be viewed as a form of regularization similar to stochastic denoising. In Bayesian terms, dropout approximates variational inference with a particular prior on weights. Empirically, dropout significantly reduces overfitting on vision tasks. The dropout probability $p$ is a tunable hyperparameter (common choices 0.2--0.5).

\subsection{Early Stopping}
Early stopping~\cite{Prechelt1998} terminates training when performance on a validation set stops improving. In effect, it limits the training time of iterative optimization, acting as a regularizer. This mirrors classical iterative methods: stopping gradient descent early avoids fitting noise. Early stopping was well-studied even before deep learning (see Prechelt (1998)\cite{Prechelt1998}). It is simple and effective: one simply tracks validation loss and halts training when it begins to rise. This adds no additional terms to the loss, but reduces model variance by effectively constraining model complexity (the number of gradient steps).

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.6\textwidth, xlabel=Epoch, ylabel=Loss, legend style={at={(0.5,-0.15)},anchor=north}]
\addplot [blue, thick] coordinates {
    (0,1.00) (10,0.667) (20,0.500) (30,0.400) (40,0.333) (50,0.286)
    (60,0.250) (70,0.222) (80,0.200) (90,0.182) (100,0.167)
};
\addlegendentry{Training Loss}
\addplot [red, thick, dashed] coordinates {
    (0,1.03) (10,0.696) (20,0.530) (30,0.393) (40,0.363) (50,0.315)
    (60,0.373) (70,0.433) (80,0.493) (90,0.553) (100,0.613)
};
\addlegendentry{Validation Loss}
\end{axis}
\end{tikzpicture}
\caption{Illustration of training vs.\ validation loss over epochs. Training loss decreases monotonically, but validation loss eventually rises (overfitting). Early stopping (around epoch 50 here) prevents fitting noise, analogous to truncating an iterative solver.}
\label{fig:loss-curves}
\end{figure}

\subsection{Implicit Regularization by Optimization}
An intriguing phenomenon is that even unregularized training can implicitly favor simpler solutions. For instance, stochastic gradient descent (SGD) is biased toward flat minima and simpler parameter configurations. Recent theory shows that for certain losses, gradient descent converges to minimum-norm interpolants; Soudry \emph{et al}. showed that for separable logistic regression, gradient descent finds the maximum-margin separator (smallest norm). Hardt \emph{et al}.~\cite{Hardt2016}. proved that with appropriate step sizes, SGD is algorithmically stable, which implies generalization guarantees for convex models. In nonconvex deep nets, theory is less complete, but the practice of using SGD, momentum, and dropout seems to enforce forms of simplicity not fully captured by classical norms. This is an active research area: e.g.\ ``double descent'' theory (Zhang \emph{et al}., 2017\cite{Zhang2017}) shows that overparameterized networks can generalize well even when fitting data exactly.

\subsection{Learned Iterative Schemes (Unrolling)}
Another bridge between classical and deep methods is algorithm unrolling. One can ``unroll'' an iterative solver into a neural network and learn its parameters from data. For example, LISTA (Learned ISTA) by Gregor and LeCun~\cite{Gregor2010} replaces the fixed matrices in iterative shrinkage-thresholding with learned weights, achieving faster convergence. Similarly, plug-and-play methods embed a trained denoiser (often a CNN) into classical schemes like ADMM or gradient descent: each iteration applies the denoiser as a prior step. This hybrid approach combines the data-driven power of neural nets with the structural insight of optimization. In effect, the network learns an implicit regularizer by training on example pairs, much as classical iterative methods use an explicit prior.
