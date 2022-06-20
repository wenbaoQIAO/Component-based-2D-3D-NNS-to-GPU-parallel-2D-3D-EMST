# Component-based-2D-3D-NNS-to-GPU-parallel-2D-3D-EMST
Abstract
We present improved data parallel approaches working on graphics processing unit (GPU) compute unified device architecture (CUDA) platform to build hierarchical Euclidean minimum spanning forest or tree for applications whose input only contains N points with arbitrary data distribution in 2D/3D Euclidean space (2D/3D EMSF/EMST). Characteristic of the proposed parallel algorithms follows ``data parallelism, decentralized control and O(1) local memory occupied by each GPU thread". This research has to solve GPU parallelism of component-based nearest neighbor search (component-based NNS), tree traversal, and other graph operations like union-find.
For NNS, instead of using classical K-d tree search or brute-force computing method, we propose a K-d search method working based on dividing the Euclidean K-dimensional space into congruent and non-overlapping square/cubic cells where size of points in each cell is bounded. For component-based NNS, with the combination of uniqueness property in Euclidean space and the K-d search, we propose dynamic and static pruning techniques based on 2D/3D square/cubic space partition to prune unnecessary neighbor cells' search. 
For tree traversal, instead of using breadth-first-search, this paper proposes CUDA kernels working with a distributed dynamic link list for selecting a local spanning tree's shortest outgoing edge since size of local EMSTs in EMSF can not be predicted. Source code is provided and experimental comparisons are conducted on both 2D and 3D benchmarks with up to 10^7 points to build final EMST. 
Results show that applying K-d search with static pruning technique and the proposed operators totally working in parallel on GPU, our current implementation runs faster than our previous work and current optimal dual-tree mlpack EMST library.


Details of this algorithm can be found in the paper: 

Qiao, W. B., & Créput, J. C. (2021). Component-based 2-/3-dimensional nearest neighbor search based on Elias method to GPU parallel 2D/3D Euclidean Minimum Spanning Tree Problem. Applied Soft Computing, 100, 106928. https://doi.org/10.1016/j.asoc.2020.106928
