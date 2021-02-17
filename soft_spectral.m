function out = soft_spectral(x,tau)
m = abs(x);
m_t = soft(x,tau)./x;
m_t(x==0) = 0;
out = m_t.*x;