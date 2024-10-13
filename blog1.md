## A rigourous mathematical exploration of Variational Autoencoders

Variational Autoencoders (VAEs) represent a fascinating fusion of deep learning and probabilistic modeling. This post aims to provide a comprehensive exploration of VAEs, balancing rigorous mathematics with practical examples and applications. We'll journey from foundational concepts to advanced theoretical insights, making this exploration accessible to both theoretically inclined beginners, regular readers and practitioners. We will certainly try our best to make you understand *how to use VAE in generating an image* using a toy example.

##  Probabilistic Foundations

###  Measure-Theoretic Probability

Let $(\Omega, \mathcal{F}, P)$ be a probability space, where:
- $\Omega$ is the sample space
- $\mathcal{F}$ is a $\sigma$-algebra on $\Omega$
- $P: \mathcal{F} \to [0,1]$ is a probability measure

**Toy Example:** Imagine we're modeling the process of generating handwritten digits. Here, $\Omega$ could be all possible 28x28 pixel images, $\mathcal{F}$ would be sets of these images (e.g., all images of the digit "7"), and $P$ would assign probabilities to these sets.

### Random Variables and Distributions

Let $X: \Omega \to \mathcal{X}$ and $Z: \Omega \to \mathcal{Z}$ be random variables representing observed and latent data respectively.

The core of VAE theory lies in the following decomposition:

$$
p(x, z) = p(x|z)p(z)
$$

where $p(z)$ is the prior over latent variables and $p(x|z)$ is the likelihood.

**Proof:** This follows directly from the definition of conditional probability:

$$
\begin{align*}
p(x, z) &= p(x|z)p(z) \quad \text{(by definition of conditional probability)} \\
&= p(z|x)p(x) \quad \text{(alternative factorization)}
\end{align*}
$$

**Practical Example:** In our handwritten digit VAE:
- $x$ could be a 28x28 image of a digit
- $z$ could be a 10-dimensional vector encoding factors like stroke width, tilt, etc.
- $p(z)$ might be a standard normal distribution $\mathcal{N}(0, I)$
- $p(x|z)$ would be our decoder network, generating an image given a latent code

##  The Variational Inference Framework

### The Evidence Lower Bound (ELBO)

**Theorem (Evidence Lower Bound):** For any distributions $q(z|x)$ and $p(x,z)$,
$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x,z) - \log q(z|x)] \equiv \text{ELBO}(x)
$$

**Proof:**
$$
\begin{align*}
\log p(x) &= \log \int p(x,z) dz \\
&= \log \int p(x,z) \frac{q(z|x)}{q(z|x)} dz \\
&= \log \mathbb{E}_{q(z|x)}\left[\frac{p(x,z)}{q(z|x)}\right] \\
&\geq \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] \quad \text{(by Jensen's inequality)}
\end{align*}
$$

**Practical Implementation:** In PyTorch, we might implement the ELBO loss as:

```python
def elbo_loss(x, x_recon, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
```

###  ELBO Decomposition

We can further decompose the ELBO:

$$
\text{ELBO}(x) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) || p(z))
$$

**Proof:**
$$
\begin{align*}
\text{ELBO}(x) &= \mathbb{E}_{q(z|x)}[\log p(x,z) - \log q(z|x)] \\
&= \mathbb{E}_{q(z|x)}[\log p(x|z) + \log p(z) - \log q(z|x)] \\
&= \mathbb{E}_{q(z|x)}[\log p(x|z)] + \mathbb{E}_{q(z|x)}[\log p(z) - \log q(z|x)] \\
&= \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) || p(z))
\end{align*}
$$

**Application:** In image generation, this decomposition shows us we're balancing two objectives:
1. Reconstruction accuracy: How well can we reconstruct the input image?
2. Latent space regularity: How close is our encoded distribution to the prior?

## The Reparameterization Trick

**Theorem  (Reparameterization Trick):** Let $g_\phi(\epsilon, x)$ be a differentiable transformation. Then,
$$
\mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{\epsilon \sim p(\epsilon)}[f(g_\phi(\epsilon, x))]
$$
where $\epsilon$ is drawn from some fixed distribution $p(\epsilon)$.

**Proof:** Let $z = g_\phi(\epsilon, x)$ where $\epsilon \sim p(\epsilon)$. Then,
$$
\begin{align*}
\mathbb{E}_{q_\phi(z|x)}[f(z)] &= \int f(z) q_\phi(z|x) dz \\
&= \int f(g_\phi(\epsilon, x)) p(\epsilon) d\epsilon \quad \text{(by change of variables)} \\
&= \mathbb{E}_{\epsilon \sim p(\epsilon)}[f(g_\phi(\epsilon, x))]
\end{align*}
$$

**PyTorch Implementation:**

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

##  Advanced ELBO Analysis

**Theorem  (Information-Theoretic ELBO Decomposition):** The ELBO can be expressed as:
$$
\text{ELBO} = \mathbb{E}_{p_\text{data}(x)}[\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]] - I_q(x;z) + H(x)
$$
where $I_q(x;z)$ is the mutual information between $x$ and $z$ under $q_\phi(x,z) = p_\text{data}(x)q_\phi(z|x)$, and $H(x)$ is the entropy of the data distribution.

**Proof:**
$$
\begin{align*}
\text{ELBO} &= \mathbb{E}_{p_\text{data}(x)}[\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) || p(z))] \\
&= \mathbb{E}_{p_\text{data}(x)}[\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]] - \mathbb{E}_{p_\text{data}(x)}[\text{KL}(q_\phi(z|x) || p(z))] \\
&= \mathbb{E}_{p_\text{data}(x)}[\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]] - (I_q(x;z) + \mathbb{E}_{p_\text{data}(x)}[\text{KL}(q_\phi(z|x) || q_\phi(z))]) \\
&= \mathbb{E}_{p_\text{data}(x)}[\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]] - I_q(x;z) + H(x)
\end{align*}
$$

**Application:** This decomposition reveals insights for disentangled representation learning. By explicitly controlling $I_q(x;z)$, we can encourage the model to learn more interpretable latent factors.

##  Normalizing Flows and VAEs

**Theorem  (Change of Variables Formula for Normalizing Flows):** Let $z_K = f_K \circ ... \circ f_1(z_0)$ be a normalizing flow. Then,
$$
\log q_K(z_K) = \log q_0(z_0) - \sum_{k=1}^K \log \left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|
$$

**Proof:** By the change of variables formula and the chain rule:
$$
\begin{align*}
q_K(z_K) &= q_0(z_0) \prod_{k=1}^K \left|\det \frac{\partial f_k^{-1}}{\partial z_k}\right| \\
\log q_K(z_K) &= \log q_0(z_0) + \sum_{k=1}^K \log \left|\det \frac{\partial f_k^{-1}}{\partial z_k}\right| \\
&= \log q_0(z_0) - \sum_{k=1}^K \log \left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|
\end{align*}
$$

**Practical Example:** Implementing a simple planar flow in PyTorch:

```python
class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))
        self.u = nn.Parameter(torch.randn(dim))

    def forward(self, z):
        linear_term = torch.sum(self.w * z, dim=1, keepdim=True) + self.b
        return z + self.u * torch.tanh(linear_term)

    def log_det_jacobian(self, z):
        linear_term = torch.sum(self.w * z, dim=1, keepdim=True) + self.b
        psi = (1 - torch.tanh(linear_term)**2) * self.w
        return torch.log(torch.abs(1 + torch.sum(psi * self.u, dim=1, keepdim=True)))
```

##  Tighter Variational Bounds

**Theorem  (IWAE Bound):** For any number of samples $K$,
$$
\log p(x) \geq \mathbb{E}_{z_1,...,z_K \sim q_\phi(z|x)} \left[\log \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}\right] \equiv \mathcal{L}_K
$$

**Proof:** By Jensen's inequality and the concavity of log:
$$
\begin{align*}
\log p(x) &= \log \mathbb{E}_{z \sim q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right] \\
&= \log \mathbb{E}_{z_1,...,z_K \sim q_\phi(z|x)}\left[\frac{1}{K} \sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}\right] \\
&\geq \mathbb{E}_{z_1,...,z_K \sim q_\phi(z|x)} \left[\log \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}\right]
\end{align*}
$$

**PyTorch Implementation:**

```python
def iwae_loss(x, model, K=10):
    x_expanded = x.unsqueeze(0).expand(K, *x.shape)
    x_recon, mu, logvar = model(x_expanded)
    
    z = model.reparameterize(mu, logvar)
    log_p_x_given_z = -F.mse_loss(x_recon, x_expanded, reduction='none').sum(dim=(1,2,3))
    log_p_z = -0.5 * z.pow(2).sum(dim=1)
    log_q_z_given_x = -0.5 * (logvar + (z - mu).pow(2) / logvar.exp()).sum(dim=1)
    
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
    return -torch.logsumexp(log_w, dim=0) + np.log(K)
```


##  Mathematical Model of a VAE for Image Generation

Let's develop a rigorous mathematical model for a VAE designed to generate images, accompanied by a toy example to illustrate the concepts.

###  Data and Latent Space

Let $\mathcal{X} = [0, 1]^{n^2}$ be our data space, representing all possible $n \times n$ grayscale images. Each $x \in \mathcal{X}$ is a flattened vector of pixel intensities.

Let $\mathcal{Z} = \mathbb{R}^d$ be our latent space, where $d$ is the dimensionality of the latent representation.

**Toy Example:**
Let's consider a very simple case where:
- Our images are 3x3 grayscale (n = 3)
- Our latent space is 2-dimensional (d = 2)

So, $\mathcal{X} = [0, 1]^9$ and $\mathcal{Z} = \mathbb{R}^2$

A sample image might look like:
```
[0.1, 0.5, 0.9]
[0.2, 0.6, 0.8]
[0.3, 0.7, 0.4]
```
Which we'd flatten to: x = [0.1, 0.5, 0.9, 0.2, 0.6, 0.8, 0.3, 0.7, 0.4]

###  Generative Model

We define our generative model as follows:

1. Prior: $p(z) = \mathcal{N}(0, I_d)$, where $I_d$ is the $d$-dimensional identity matrix.

2. Decoder: $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I_{n^2})$, where:
   - $\mu_\theta: \mathbb{R}^d \to [0, 1]^{n^2}$ is a neural network with parameters $\theta$
   - $\sigma^2$ is a fixed variance for simplicity (though this could also be learned)

**Toy Example:**
Let's define a simple decoder network:

$\mu_\theta(z) = \text{sigmoid}(W_2 \cdot \text{ReLU}(W_1z + b_1) + b_2)$

Where:
$W_1 \in \mathbb{R}^{4 \times 2}, b_1 \in \mathbb{R}^4, W_2 \in \mathbb{R}^{9 \times 4}, b_2 \in \mathbb{R}^9$

Let $\sigma^2 = 0.1$

###  Inference Model

Our inference model (encoder) is defined as:

$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$

where:
- $\mu_\phi: [0, 1]^{n^2} \to \mathbb{R}^d$ is a neural network with parameters $\phi$
- $\Sigma_\phi: [0, 1]^{n^2} \to \mathbb{R}^{d \times d}$ is a neural network outputting a diagonal covariance matrix

**Toy Example:**
Let's define a simple encoder network:

$[\mu_\phi(x), \log \Sigma_\phi(x)] = W_4 \cdot \text{ReLU}(W_3x + b_3) + b_4$

Where:
$W_3 \in \mathbb{R}^{6 \times 9}, b_3 \in \mathbb{R}^6, W_4 \in \mathbb{R}^{4 \times 6}, b_4 \in \mathbb{R}^4$

The output is split into $\mu_\phi(x) \in \mathbb{R}^2$ and $\log \Sigma_\phi(x) \in \mathbb{R}^2$ (diagonal elements of the covariance matrix in log space).

### 7.4 ELBO for Image Generation

The Evidence Lower Bound (ELBO) for our image generation VAE is:

$$
\text{ELBO}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) || p(z))
$$

Let's expand each term:

1. Reconstruction term:

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] &= \mathbb{E}_{q_\phi(z|x)}\left[-\frac{n^2}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}||x - \mu_\theta(z)||^2\right] \\
&\approx -\frac{n^2}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}||x - \mu_\theta(\mu_\phi(x))||^2
\end{aligned}
$$

where we've used a single sample approximation for the expectation.

2. KL divergence term:

$$
\text{KL}(q_\phi(z|x) || p(z)) = \frac{1}{2}\left(\text{tr}(\Sigma_\phi(x)) + \mu_\phi(x)^T\mu_\phi(x) - d - \log \det(\Sigma_\phi(x))\right)
$$

**Toy Example Calculation:**
Let's calculate the ELBO for our sample image x = [0.1, 0.5, 0.9, 0.2, 0.6, 0.8, 0.3, 0.7, 0.4]

1. Encode: 
   Suppose $\mu_\phi(x) = [0.5, -0.3]$ and $\log \Sigma_\phi(x) = [-1.2, -0.8]$

2. Sample z:
   $z = [0.4, -0.5]$ (sampled from $\mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$)

3. Decode:
   Suppose $\mu_\theta(z) = [0.15, 0.45, 0.85, 0.25, 0.55, 0.75, 0.35, 0.65, 0.45]$

4. Reconstruction term:
   $$-\frac{9}{2}\log(2\pi \cdot 0.1) - \frac{1}{2 \cdot 0.1}(0.1-0.15)^2 + ... + (0.4-0.45)^2 \approx -10.5$$

5. KL divergence term:
   $$\frac{1}{2}(e^{-1.2} + e^{-0.8} + 0.5^2 + (-0.3)^2 - 2 + 1.2 + 0.8) \approx 0.3$$

6. ELBO:
   $$\text{ELBO}(x) \approx -10.5 - 0.3 = -10.8$$

###  Optimization Objective

Our final optimization objective is to maximize the ELBO:

$$
\max_{\theta, \phi} \mathbb{E}_{p_\text{data}(x)}[\text{ELBO}(x)]
$$

which is equivalent to minimizing the negative ELBO:

$$
\min_{\theta, \phi} \mathbb{E}_{p_\text{data}(x)}\left[\frac{1}{2\sigma^2}||x - \mu_\theta(\mu_\phi(x))||^2 + \frac{1}{2}\left(\text{tr}(\Sigma_\phi(x)) + \mu_\phi(x)^T\mu_\phi(x) - d - \log \det(\Sigma_\phi(x))\right)\right]
$$

**Toy Example:**
In practice, we would use stochastic gradient descent to optimize this objective over a dataset of images. For our single image example, one step of gradient descent might look like:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \text{ELBO}(x)
$$
$$
\phi \leftarrow \phi + \alpha \nabla_\phi \text{ELBO}(x)
$$

Where $\alpha$ is the learning rate.

###  Image Generation Process

To generate a new image:

1. Sample $z \sim \mathcal{N}(0, I_d)$
2. Compute $\mu_\theta(z)$
3. Sample $x \sim \mathcal{N}(\mu_\theta(z), \sigma^2 I_{n^2})$
4. Reshape the resulting vector into an $n \times n$ image

**Toy Example:**
1. Sample $z = [0.1, 0.7]$ from $\mathcal{N}(0, I_2)$
2. Compute $\mu_\theta(z) = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.1, 0.8, 0.9]$
3. Sample $x$ from $\mathcal{N}(\mu_\theta(z), 0.1 I_9)$, say we get:
   $x = [0.25, 0.35, 0.55, 0.35, 0.45, 0.75, 0.15, 0.85, 0.95]$
4. Reshape into a 3x3 image:
   ```
   [0.25, 0.35, 0.55]
   [0.35, 0.45, 0.75]
   [0.15, 0.85, 0.95]
   ```

###  Theoretical Properties

1. **Consistency**: As the number of training samples approaches infinity and with sufficient model capacity, the VAE converges to the true data distribution.

2. **Manifold Learning**: The VAE learns a low-dimensional manifold in $\mathcal{Z}$ that captures the structure of the image data.

3. **Disentanglement**: Under certain conditions (e.g., using a β-VAE formulation), the latent space can capture disentangled factors of variation in the images.

4. **Sample Quality vs. Reconstruction Quality Trade-off**: There's an inherent tension between generating high-quality samples and accurately reconstructing inputs, controlled by the weight of the KL divergence term.

**Toy Example Illustration:**
In our 2D latent space, we might find that:
- One dimension controls the overall brightness of the image
- The other dimension controls the contrast between the center and edges

This would be an example of learning a disentangled representation, where each latent dimension corresponds to a meaningful factor of variation in the data.

## Conclusion

Through this detailed exploration of VAEs for image generation, complete with a toy example, we've seen how these models bridge the gap between deep learning and probabilistic modeling. The mathematical formulation provides a rigorous foundation, while the toy example offers concrete insights into how VAEs operate in practice.

This combination of theory and example illustrates the power of VAEs in learning compact, meaningful representations of complex data like images. As we continue to advance both the theory and application of VAEs, we open up new possibilities in generative modeling, representation learning, and beyond.
