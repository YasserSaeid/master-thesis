\chapter{System Model and Problem Description}
\label{cha:prob_d}
By deploying a large antenna array at the base
station (BS), the base station is able to perform both receive and transmit beamforming with
narrow beams, thus eliminating multiuser interference and thereby increasing the cell throughput.
For effective downlink beamforming, it is essential to have accurate knowledge of the downlink
channel state information at the transmitter (CSIT). In a time-division duplexing (TDD) system,
downlink CSI can be obtained by exploiting uplink/downlink channel reciprocity. The common
assumption in Massive MIMO is that each user terminal (UT) only has a small number of
antennas and that the BS can use uplink channel information, obtained from the relatively
easy uplink training, for downlink beamforming  \cite{CE_FDD_ref2}. On the other hand, as frequency-division
duplexing (FDD) is generally considered to be more effective for systems with symmetric traffic
and delay-sensitive applications, most cellular systems today employ FDD \cite{CE_FDD_ref2} \cite{CE_FDD_ref5}. Channel
reciprocity is no longer valid in FDD systems and in order to obtain CSIT, the BS has to
perform downlink training. Subsequently, the user needs to estimate, quantize and feedback the
channel state information.

To deal with the limited downlink channel training interval in a FDD Massive MIMO system,
one option is to explore the possible underlying channel structure whereby the high dimensional
channel vector has a low dimensional representation \cite{CE_FDD_REF_3} \cite{CE_FDD_REF6}. Motivated by the framework of
Compressive Sensing (CS), if the desired signal (channel response) can be sparsely represented
in some basis or dictionary, then the number of measurements (downlink training period) is
proportional to the number of nonzero entries and the signal can be robustly recovered using
sparse signal recovery algorithms \cite{CE_FDD_REF_7}\cite{CE_FDD_REF_8}. 

\section{Sparse Channel Representation}
The insight behind the dictionary learning approach is that
the high dimensional data (channel response in our case) usually has some structure correlated
in some dimensions, and the true degrees of freedom that generate the data is usually small. So
by learning from large amount of data, we are able to recover useful underlying structures or
models, thus making representation of the data more efficient for the desired application. 

Consider a single cell downlink multiuser MIMO system
with $N$ antennas at the base station and $K$ single-antenna
users. The motivating assumption underneath the dictionary-learning based approach is that the channel vectors $\mathbf{h}_k\in \mathbb{C}^{N\times 1}$, $k\in [K]$, admit a \textit{sparse} representation with respect to a dictionary $\mathbf{C}$, i.e. 

\begin{equation}
\mathbf{h}_k=\mathbf{C}\mathbf{x}_k, \:\: \mathrm{where} \Vert\mathbf{x}_k\Vert_{0}< N.
\label{eq:channel_representation}
\end{equation} 

In this regard, there are two important 
questions: first, does the underlying physical propagation environment support a sparse representation?
Second, can we find a good sparsifying dictionary $\mathbf{C}$ 
such that the mathematical model is valid?

The sparse channel representation is supported by the observations made in \cite{CE_FDD:journals/corr/DingR16}, \cite{CE_FDD_REF4}, \cite{GSCM}, \cite{CE_FDD_REF_42} for the Geometry-Based Stochastic Channel Model (GSCM) \cite{GSCM}. To summarize, the assumptions are that for a specific cell, the locations of the dominant scattering clusters are determined by cell specific attributes such as the buildings, terrain, etc. and are common to all the users
irrespective of user position. Given that the scattering clusters are far away from the base
station, the subpaths associated with a specific scattering cluster will be concentrated in a
small range around the line of sight (LOS) direction between the base station and the scattering
cluster, i.e. having a small angular spread (AS). Further, by assuming that the user terminals are far away from the base station,
subpaths associated with the user-location dependent scattering cluster also have small angular
spread. For each link between the base station and the user, the number of scattering clusters
that contributes to the channel responses is typically small. For users
at different locations, if they can see the same scattering cluster then their channel responses
will contain subpaths with similar AOA/AOD which all concentrate around the LOS direction
between the base station and that scattering cluster, a phenomenon known as ”shared scatterers” 
or ”joint scatterers”. These considerations, if valid, support the idea of a low dimensional
representation for the large Massive MIMO channel. 

\section{Dictionary Learning for Channel Representation}

Dictionary learning in the context of channel estimation in FDD massive MIMO systems has been addressed in \cite{CE_FDD:journals/corr/DingR16}. Specifically, in \cite{CE_FDD_REF4}, \cite{CE_FDD_REF6} and \cite{CE_FDD_REF_9} the discrete Fourier transform (DFT) matrix has been used to
sparsely represent the channel. The utilization of DFT basis is compatible with
theoretical results of signal recovery in compressive sensing \cite{CE_FDD_REF_7}, and has been proposed as the
virtual channel model  \cite{CE_FDD_REF_10} or angular domain channel representation  \cite{CE_FDD_REF_11}. However, the DFT
basis is only valid for a uniform linear array (ULA), and can only lead to an approximate sparse
representation with limited scattering and sufficiently large number of antennas  \cite{CE_FDD_REF_9},\cite{CE_FDD_REF_10}. For
practical channels, the DFT basis will often result in a large number of nonzero entries in the
channel representation, which in turn requires a large number of training symbols for reliable
channel estimation, thus losing the benefits of CS based channel estimation.

%Motivated by the CS framework, this work extends the framework to provide effective solutions
%to the FDD Massive MIMO channel estimation problem. 
In order to accurately and sparsely
represent the channel, a dictionary learning based channel model (DLCM) was proposed in \cite{CE_FDD:journals/corr/DingR16}, where a
learned overcomplete dictionary, rather than a predefined basis or dictionary, is used to represent
the channel. The dictionary, due to the learning process, is able to adapt to the cell characteristics
as well as insure a sparse representation of the channel. Since no structural constraints are placed
on the dictionary, the approach is applicable to an arbitrary array geometry and also does not need
accurate array calibration. Being cell specific, a further obvious advantage of the dictionary learning process is that it adapts the channel model to the 
channel measurements collected in a cell. The
sparse representation is also encouraged during the learning process. The learned overcomplete dictionary has the
potential to exploit underlying low dimensionality through the learning process, and is robust
to antenna array uncertainties and non-ideal propagation schemes. In particular, as the learned dictionary does not have any predefined
structural constraints, it is robust towards towards imperfections in the underlying physical generation scheme of the channel, and is expected to also work in the case when antenna gains and locations are different from the nominal values, or there exist near-field
scattering clusters.  

The dictionary-learning problem has been formalized in \cite{CE_FDD:journals/corr/DingR16}. For completeness, we provide a summary in the following. %From now on, we denote $\mathbf{D}^d \in \mathbb{C}^{
%N\times M}$ as the dictionary that has to be learned from downlink channel measurements. To benefit from the flexibility of overcompleteness, we let $N < M$. 
Assuming we collect $L$ channel measurement vectors as training samples in a specific cell, the goal is to learn the dictionary $\mathbf{C} \in \mathbb{C}^{N\times M}$ (where $N<M$ to benefit from overcompleteness), 
such that for all channel responses $\mathbf{h}_i, i=1,\ldots,L$, they can be approximated as $\mathbf{h}_i\approx \mathbf{C}\mathbf{x}_i$, $\mathbf{x}_i\in \mathbb{C}^{M\times 1}$. The dictionary learning algorithm should in general be able to address model fitting $\Vert \mathbf{h}_i-\mathbf{C}\mathbf{x}_i\Vert_2$
(accuracy), and encourage small $\Vert \mathbf{x}_i\Vert_0$ (efficiency) for sparse representation. If one constrains the 
model mismatch error, then the dictionary learning problem can be formulated as
\begin{align}
&\min\limits_{\mathbf{x}_1,\ldots,\mathbf{x}_L} \frac{1}{L}\sum_{i=1}^L\Vert\mathbf{x}_i\Vert_0\\
&\mathrm{s.t.}\:\:\Vert\mathbf{h}_i-\mathbf{C}\mathbf{x}_i\Vert_2\leq \eta,\:\forall i
\label{eq:DL_vector_form}
\end{align}
and $\Vert \mathbf{C}_{.j}\Vert_2\leq 1$, $\forall j=1,\ldots, M$ in order to prevent ambiguity between $\mathbf{C}$ and $\mathbf{x}$. The solved $\mathbf{C}$ in (\ref{eq:DL_vector_form}) leads to the sparsest representation in the sense of all channel measurements, given the model mismatch tolerance $\eta$.

Two similar formulations could alternatively be used. If one knows beforehand or wants to
constrain the sparsity level of each coefficient $\mathbf{x}_i$, then one solves:
\begin{align}
&\min\limits_{\mathbf{x}_1,\ldots,\mathbf{x}_L} \frac{1}{L}\sum_{i=1}^L\frac{1}{2}\Vert\mathbf{h}_i-\mathbf{C}\mathbf{x}_i\Vert_2^2\\
&\mathrm{s.t.}\:\:\Vert\mathbf{x}_i\Vert_0\leq T_0,\:\forall i
\label{eq:DL_vector_form_2}
\end{align}
where $T_0$ constrains the number of non-zero elements in each $\mathbf{x}_i$
. In other words, one expects that 
every channel measurement can be represented using $T_0$ atoms from the learned dictionary, and
one solves for the dictionary that minimizes model mismatch using channel response samples. If no explicit constraints are posed on the model fitting error or sparsity level, the dictionary learning process can be formulated in a general form as
\begin{align}
&\min\limits_{\mathbf{x}_1,\ldots,\mathbf{x}_L} \frac{1}{L}\sum_{i=1}^L\frac{1}{2}\Vert\mathbf{h}_i-\mathbf{C}^d\mathbf{x}_i\Vert_2^2+\lambda\Vert\mathbf{x}_i\Vert_0
\label{eq:DL_vector_form_3}
\end{align}
where $\lambda$ is the parameter that trades off the data fitting and sparsity.
To solve the dictionary learning problem, in \ref{sect:DLCM} block coordinate descent framework
has been applied where each iteration includes alternatively minimizing with respect to
either $\mathbf{C}$ or $\mathbf{x}_i$, $\forall i$
, while keeping the other fixed. In \ref{sect:DLCM} it has been shown experimentally that starting from a reasonable
dictionary, e.g. an overcomplete DFT dictionary, the learning algorithm will lead to a dictionary
that improves the performance in terms of both sparse representation and sparse recovery.

\section{Dictionary Learning for Channel Estimation}

Once the channel has the sparse representation, the compressed
channel estimation framework can be utilized to reduce the amount of downlink training. Here we address a particular implementation of downlink channel estimation with \textit{explicit feedback}. According to this approach, the users feed back (quantized) received training signals to the base station and channel is recovered at the base station. This is different from the conventional channel estimation schemes where the users estimate the channel (or calculate precoding matrices), and report feedback to the base station in the form of channel quality indicator(CQI) and precoding matrix indicator (PMI) (as e.g. in LTE). Such scheme has several advantages: firstly the sparse recovery process requires complicated computation so it is preferably done at the base station thus saving energy for user terminals. Secondly, the dictionary is learned and stored at the base station, making it available to all users which otherwise involves significant overhead in storage at the user equipment and conveyance of dictionary. Furthermore, such overhead will be incurred every time user enters a new cell. By assigning the recovery operation to the base station, one avoids such overhead. In addition, since dimension of received symbol $T$ is less than the channel dimension $N$ in a Massive MIMO system, the scheme also reduces feedback overhead. 

%\section{Dictionary Learning for User Grouping}

%The concept of dictionary learning can also be used for user grouping in the context of the 
%Joint-Spatial-and-Division-Multiplexing (JSDM) approach
%to multiuser MIMO downlink. It works
%by partitioning users with the same second order downlink
%channel statistics into groups and splitting the downlink beamforming
%into two stages: an outer precoder, that depends on
%the channel statistics, and an inner precoder that depends on
%the instantaneous effective channel realizations. The role of the
%precoders is to suppress inter-group and intra-group interference
%respectively. The dimensions of the effective channel
%is significantly less than the number of antennas, thanks to the
%outer precoder projection. Even with this reduction in CSIT
%feedback, the authors in \cite{Assaad_4} showed that JSDM achieves the
%same sum capacity of the corresponding MU-MIMO downlink
%channel if the eigenspaces of groups are mutually orthogonal,
%a condition they called ”tall unitary”. In realistic scenarios, users might have similar but not
%necessarily identical second order downlink channel statistics.
%This dictates the incorporation of a clustering algorithm to
%partition users into groups with sufficiently similar covariance
%eigenspaces. On top of that, with a high number of users
%uniformly distributed across the cell, the eigenspaces of the
%groups are far from meeting the tall unitary condition and
%a reduction of the number of simultaneously served groups
%is required.






%+Massive Multiple-Input-Multiple-Output (MIMO) is introduced as the next generation of communication systems. What makes massive MIMO systems very useful, is the ability to deploy a huge number of antennas at the base station side. This ability enables BSs to perform both receive and transmit beamforming. The downlink beamforming (BS to MS) requires a preparatory knowledge about the channel channel information (CSI). That is the case in FDD massive MIMO systems. Frequency-division duplexing massive MIMO systems operate using two different channels (one for uplink and another one for the downlink). In order to obtain proper channel status information, a MS needs to estimate, quantize and feedback CSI to the BS. Because of the huge number deployment of antennas in FFD MIMO system, it is difficult to perform downlink training and then feedbacking the CSI to BS. The channel status information is probational to the number of antennas which is huge in our case. \\
%In this paper, we try to solve this problems by searching for a lower dimensional representation of the channel. Inspired by the Compressed Sensing framework, we try to find a signal which has a sparse representation in a certain dictionary. The sparse representation of the downlink channel is advantageous to us, due to the fact that the downlink training period is , in this case, proportional to only few non-zero elements.  That allows us to robustly estimate the downlink channel.Hence we introduce in this study a \textit{method dictionary learning based channel model} to find a certain basis (dictionary).
%In massive MIMO systems, the downlink estimation is not easy task to achieve while the uplink estimation is an easy one to perform thanks to the small number of antenna at a MS side.
%An another challenge that MSs face when users need to estimate the downlink channel during a short downlink training period.





\noindent 

\newpage\null\thispagestyle{empty}\newpage
