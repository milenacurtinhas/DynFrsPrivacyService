// #include <bits/stdc++.h>
// #include <omp.h>
#include <algorithm>
#include <unordered_map>
#include <chrono>

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// input data dtype
typedef double Tx;

// input label dtype
typedef int Ty;

// split score dtype
typedef double Ts;

const Tx eps = 1e-8;

random_device rd;
mt19937 mt(rd());

int randint(int l, int r) {
	return rand() % (r - l + 1) + l;
}

vector<int> id_T;
vector<int> id_d;

bool dly = 1;
int p_tries = 20;
int p_count = 20;

int counter = 0;

bool erase(vector<int> &vec, int a) {
	for (int i = vec.size() - 1; i >= 0; --i) if (vec[i] == a) {
		swap(vec[i], vec.back());
		vec.pop_back();
		return 1;
	}
	return 0;
}

Ts calc_score(int ls_s, int ls_1, int rs_s, int rs_1) {
	return ((Ts) (ls_s - ls_1) * ls_1 / ls_s + (Ts) (rs_s - rs_1) * rs_1 / rs_s) / (ls_s + rs_s);
}

// specialized for binary classification
class random_forest {
public:
	
	int d;
	int n;
	int C;
	vector<vector<Tx>> X;
	vector<Ty> Y;
	vector<bool> X_binary;
	vector<vector<int>> at;
	
	int T;
	int k;
	
	class decision_tree {
		public:
		
		const random_forest &RF;

		int max_dep;
		int min_split_size;

		vector<pair<Tx, Ty>> a;
		
		mt19937 mt;
		
		class attribute {
		public:
			// Não é necessário serializar a classe attribute pois é reconstruida automaticamente ao restaurar a árvore
		
			decision_tree &DT;

			int d, n, n_1, sz, used_count;
			vector<int> idx;
			vector<bool> used, cons;
			vector<pair<Tx, int>> mn, mx;
			vector<int> fm;
			vector<pair<Ts, Tx>> spl;
			vector<vector<Tx>> thr;
			vector<vector<pair<int, int>>> cnt;

			attribute(decision_tree &DT, int d):
				DT(DT), d(d), sz(0), n_1(0), used_count(0),
				used(vector<bool>(d, 0)), cons(vector<bool>(d, 0)),
				mn(vector<pair<Tx, int>>(p_count)),
				mx(vector<pair<Tx, int>>(p_count)),
				fm(vector<int>(p_count, -1)), spl(vector<pair<Ts, Tx>>(p_count)),
				thr(vector<vector<Tx>>(p_count)), cnt(vector<vector<pair<int, int>>>(p_count)) {
			}
			attribute(decision_tree &DT, int d, const vector<bool> &cons):
				DT(DT), d(d), sz(0), n_1(0), used_count(0),
				used(vector<bool>(d, 0)), cons(cons),
				mn(vector<pair<Tx, int>>(p_count)),
				mx(vector<pair<Tx, int>>(p_count)),
				fm(vector<int>(p_count, -1)), spl(vector<pair<Ts, Tx>>(p_count)),
				thr(vector<vector<Tx>>(p_count)), cnt(vector<vector<pair<int, int>>>(p_count)) {
			}
			
			int new_idx() {
				int at = -1;
				if (idx.size()) {
					at = idx.back();
					idx.pop_back();
					return at;
				}
				at = sz++;
				return at;
			}
			
			int get_next() {
				std::uniform_int_distribution<int> rnd(0, d - 1);
				int id = -1;
				
				for (int trial = 0; trial < p_count; ++trial) {
					id = rnd(DT.mt);
					
					// cerr << id << ": " << used[id] << ' ' << cons[id] << endl;
					
					if (used[id] | cons[id]) {
						id = -1;
						continue;
					}
					break;
				}
				// cerr << "return: " << id << endl;
				return id;
			}

			void set_as(int id, int sta) {
				// cerr << "set " << id << " as " << sta << endl;
				if (sta == 1) used[id] = 1;
				else if (sta == 2) cons[id] = 1;
			}

			pair<Ts, Tx> gen_bin(int id, int rs_s, int rs_1) {
				int at = new_idx();
				fm[at] = id;
				used_count += 1;
				used[id] = 1;
				
				auto &thr = this->thr[at];
				auto &cnt = this->cnt[at];
				auto &spl = this->spl[at];
				
				const int ls_s = n - rs_s, ls_1 = n_1 - rs_1;
				mn[at] = {0, ls_s}, mx[at] = {1, rs_s};
				
				thr.resize(1), cnt.resize(1);
				thr[0] = eps;
				cnt[0] = {ls_s, ls_1};
				spl = {calc_score(ls_s, ls_1, rs_s, rs_1), eps};
				return spl;
			}
			
			pair<Ts, Tx> gen(int id, int spl_cnt, int a_size, const vector<pair<Tx, Ty>> &a) {
				int at = new_idx();
				fm[at] = id;
				used_count += 1;
				used[id] = 1;
				
				Tx &mn = this->mn[at].first, &mx = this->mx[at].first;
				int &mn_cnt = this->mn[at].second, &mx_cnt = this->mx[at].second;
				mn = mx = a[0].first, mn_cnt = mx_cnt = 1;
				for (int i = 1; i < a_size; ++i) {
					const auto &X = a[i].first;
					if (X < mn) mn = X, mn_cnt = 1;
					else if (X > mx) mx = X, mx_cnt = 1;
					else if (X == mn) ++mn_cnt;
					else if (X == mx) ++mx_cnt;
				}
				
				auto &thr = this->thr[at];
				auto &cnt = this->cnt[at];
				auto &spl = this->spl[at];
				
				thr.resize(spl_cnt), cnt.resize(spl_cnt);
				
				uniform_real_distribution<> rnd(mn, mx);
				for (int i = 0; i < spl_cnt; ++i) thr[i] = rnd(DT.mt);
				sort(thr.begin(), thr.end());
				
				vector<int> pre_s(spl_cnt + 1, 0), pre_1(spl_cnt + 1, 0);
				for (int i = 0; i < a_size; ++i) {
					const auto &X = a[i].first;
					int pos = upper_bound(thr.begin(), thr.end(), X) - thr.begin();
					++pre_s[pos];
					if (a[i].second) ++pre_1[pos];
				}
				for (int i = 1; i < thr.size(); ++i) pre_s[i] += pre_s[i - 1], pre_1[i] += pre_1[i - 1];
				
				for (int i = 0; i < thr.size(); ++i) {
					const auto &thres = thr[i];
					int ls_s = pre_s[i], ls_1 = pre_1[i];
					Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
					cnt[i] = {ls_s, ls_1};
					
					if (!i || score < spl.first) spl = {score, thres};
				}
				return spl;
			}

			void destroy(int at) {
				int id = fm[at];
				auto &thr = this->thr[at];
				auto &cnt = this->cnt[at];
				auto &spl = this->spl[at];
				
				idx.push_back(at);
				
				// erase(used, id);
				// unused.push_back(id), shuffle_back();
				mn[at] = mx[at] = {0, -1};
				thr.clear();
				cnt.clear();
				
				fm[at] = -1;
				
				used_count -= 1;
				used[id] = 0;
			}
			
			// void ins_once(const Ty &Y) {  }
			bool add(const vector<Tx> &Xs, const Ty &Y) {
				n += 1, n_1 += Y;
				
				int destroy_cnt = 0;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					int id = fm[at];
					bool f = 0;
					const auto &X = Xs[id];
					Tx &mn = this->mn[at].first, &mx = this->mx[at].first;
					int &mn_cnt = this->mn[at].second, &mx_cnt = this->mx[at].second;
					if (X < mn) mn = X, mn_cnt = f = 1;
					else if (X > mx) mx = X, mx_cnt = f = 1;
					else if (X == mn) ++mn_cnt;
					else if (X == mx) ++mx_cnt;

					if (!f) {
						auto &thr = this->thr[at];
						auto &cnt = this->cnt[at];
						auto &spl = this->spl[at];
						for (int i = 0; i < thr.size(); ++i) {
							const auto &thres = thr[i];
							int &ls_s = cnt[i].first, &ls_1 = cnt[i].second;
							if (X <= thres) ++ls_s, ls_1 += Y;
							Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
							if (score < spl.first) spl = {score, thres};
						}
					} else {
						destroy(at);
						destroy_cnt += 1;
					}
				}
				return destroy_cnt;
			}

			int del(const vector<Tx> &X, const Ty &Y) {
				n -= 1, n_1 -= Y;
				
				int destroy_cnt = 0;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					int id = fm[at];
					bool f = 0;
					const auto &X_ = X[id];
					pair<Tx, int> &mn = this->mn[at], &mx = this->mx[at];
					if (X_ == mn.first) --mn.second, f = !mn.second;
					else if (X_ == mx.first) --mx.second, f = !mx.second;
					
					if (!f) {
						auto &thr = this->thr[at];
						auto &cnt = this->cnt[at];
						auto &spl = this->spl[at];
						for (int i = 0; i < thr.size(); ++i) {
							const auto &thres = thr[i];
							int &ls_s = cnt[i].first, &ls_1 = cnt[i].second;
							if (X_ <= thres) --ls_s, ls_1 -= Y;
							Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
							if (score < spl.first) spl = {score, thres};
						}
					} else {
						destroy(at);
						destroy_cnt += 1;
					}
				}
				
				return destroy_cnt;
			}
			
			int del(const vector<const vector<Tx>*> &Xs, const vector<Ty> &Ys) {
				n -= Ys.size();
				for (const auto &Y : Ys) n_1 -= Y;
				
				int destroy_cnt = 0;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					int id = fm[at];
					bool f = 0;
					pair<Tx, int> &mn = this->mn[at], &mx = this->mx[at];
					for (const auto &X : Xs) {
						const auto &X_ = (*X)[id];
						if (X_ == mn.first) --mn.second, f = !mn.second;
						else if (X_ == mx.first) --mx.second, f = !mx.second;
						if (f) break;
					}
					
					if (!f) {
						auto &thr = this->thr[at];
						auto &cnt = this->cnt[at];
						auto &spl = this->spl[at];
						
						vector<int> pre_s(thr.size() + 1, 0), pre_1(thr.size() + 1, 0);
						for (int i = 0; i < Ys.size(); ++i) {
							const auto &X_ = (*Xs[i])[id];
							int pos = upper_bound(thr.begin(), thr.end(), X_) - thr.begin();
							++pre_s[pos];
							if (Ys[i]) ++pre_1[pos];
						}
						for (int i = 1; i < thr.size(); ++i) pre_s[i] += pre_s[i - 1], pre_1[i] += pre_1[i - 1];
						
						for (int i = 0; i < thr.size(); ++i) {
							const auto &thres = thr[i];
							int &ls_s = cnt[i].first, &ls_1 = cnt[i].second;
							ls_s -= pre_s[i], ls_1 -= pre_1[i];
							Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
							
							if (score < spl.first) spl = {score, thres};
						}
					} else {
						destroy(at);
						destroy_cnt += 1;
					}
				}
				
				return destroy_cnt;
			}
			
			pair<Tx, int> find_best_spl() {
				int attr = -1;
				pair<Ts, Tx> best;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					const auto &split = spl[at];
					if (attr == -1 || split.first < best.first) best = split, attr = fm[at];
				}
				return {best.second, attr};
			}

			void reset() {
				fill(used.begin(), used.end(), 0);
				// fill(cons.begin(), cons.end(), 0);
				fill(fm.begin(), fm.end(), -1);
				sz = used_count = 0;
				idx.clear();
			}
		};

		class node {
			public:
			
			const random_forest &RF;
			decision_tree &DT;

			vector<int> Xid;
			attribute A;

			pair<Tx, int> spl = {0, -1};
			Tx thres;
			int attr;
			
			// delay = 0: Nó normal, já construído
			// delay = 1: Nó precisa ser construído completamente (build())
			// delay = 2: Nó precisa apenas separar dados e construir filhos (separate() + build() nos filhos)
			int delay;
			
			int dep;
			node *ls, *rs;
			
			bool old = 0;
			
			node(const random_forest &RF, decision_tree &DT, int dep) : 
				RF(RF), DT(DT), A(DT, RF.d), dep(dep) {
				ls = rs = nullptr;
				thres = 0.0;        
				attr = 0;           
				delay = 1;          
				spl = {0.0, 0};     
				old = 0;            
			}
			
			node(const random_forest &RF, decision_tree &DT, int dep, const vector<bool> &cons):
				RF(RF), DT(DT), A(DT, RF.d, cons), dep(dep) {
				ls = nullptr;       
				rs = nullptr;       
				thres = 0.0;        
				attr = 0;           
				delay = 1;          
				spl = {0.0, 0};     
				old = 0;            
			}

			void serialize(std::ostream &out) const {
				out.write((char*)&dep, sizeof(dep));
				
				uint32_t xid_size = static_cast<uint32_t>(Xid.size());
				out.write((char*)&xid_size, sizeof(xid_size));
				
				if (xid_size > 0) {
					out.write((char*)Xid.data(), xid_size * sizeof(int));
				}
				
				double safe_thres = std::isnan(thres) ? 0.0 : thres;
				int safe_attr = (attr >= 0 && attr < 10000) ? attr : 0;
				int safe_delay = delay;
				double safe_spl_first = std::isnan(spl.first) ? 0.0 : spl.first;
				int safe_spl_second = (spl.second >= -1 && spl.second < 10000) ? spl.second : -1;
				bool safe_old = old;
				
				out.write((char*)&safe_thres, sizeof(safe_thres));
				out.write((char*)&safe_attr, sizeof(safe_attr));
				out.write((char*)&safe_delay, sizeof(safe_delay));
				out.write((char*)&safe_spl_first, sizeof(safe_spl_first));
				out.write((char*)&safe_spl_second, sizeof(safe_spl_second));
				out.write((char*)&safe_old, sizeof(safe_old));
				
				// Serializar A.n e A.n_1 explicitamente
				out.write((char*)&A.n, sizeof(A.n));
				out.write((char*)&A.n_1, sizeof(A.n_1));
				
				bool has_ls = (ls != nullptr);
				bool has_rs = (rs != nullptr);
				out.write((char*)&has_ls, sizeof(has_ls));
				out.write((char*)&has_rs, sizeof(has_rs));
				if (has_ls) ls->serialize(out);
				if (has_rs) rs->serialize(out);
			}

			static node* deserialize(std::istream &in, decision_tree &DT, const random_forest &RF) {
				int dep_local;
				in.read((char*)&dep_local, sizeof(dep_local));
				
				uint32_t xid_size;
				in.read((char*)&xid_size, sizeof(xid_size));
				
				const uint32_t MAX_REASONABLE_SIZE = 100000;
				if (xid_size > MAX_REASONABLE_SIZE) {
					throw std::length_error("Tamanho inválido para Xid");
				}
				
				node* n = new node(RF, DT, dep_local);
				
				if (xid_size > 0) {
					n->Xid.resize(xid_size);
					in.read((char*)n->Xid.data(), xid_size * sizeof(int));
				}
				
				in.read((char*)&n->thres, sizeof(n->thres));
				in.read((char*)&n->attr, sizeof(n->attr));
				in.read((char*)&n->delay, sizeof(n->delay));
				in.read((char*)&n->spl.first, sizeof(n->spl.first));
				in.read((char*)&n->spl.second, sizeof(n->spl.second));
				in.read((char*)&n->old, sizeof(n->old));

				// Deserializar A.n e A.n_1 explicitamente
				in.read((char*)&n->A.n, sizeof(n->A.n));
				in.read((char*)&n->A.n_1, sizeof(n->A.n_1));
				
				bool has_ls = false, has_rs = false;
				in.read((char*)&has_ls, sizeof(has_ls));
				in.read((char*)&has_rs, sizeof(has_rs));
				if (has_ls) n->ls = node::deserialize(in, DT, RF);
				if (has_rs) n->rs = node::deserialize(in, DT, RF);
				
				return n;
			}
			
			void gen_spl(int trials) {
				bool f = !Xid.size();
				if (f) collect();
				
				pair<Ts, Tx> best;
				for (; A.used_count < trials;) {
					int p = A.get_next();
					
					if (p == -1) break;
					
					if (RF.X_binary[p]) {
						bool constant = 1;
						int cnt_1 = 0, cnt_1_1 = 0;
						const int X_0 = (bool) RF.X[Xid[0]][p];
						for (int id : Xid) {
							const bool X = RF.X[id][p];
							const auto &Y = RF.Y[id];
							cnt_1 += X;
							if (X) cnt_1_1 += Y;
						}
						
						constant = cnt_1 == 0 || cnt_1 == Xid.size();
						if (constant) {
							A.set_as(p, 2);
							continue;
						}
						
						A.set_as(p, 1);
						A.gen_bin(p, cnt_1, cnt_1_1);
					} else {
						bool constant = 1;
						const Tx &X_0 = RF.X[Xid[0]][p];
						int a_size = 0;
						DT.a.reserve(Xid.size());
						for (int id : Xid) {
							const auto &X = RF.X[id][p];
							const auto &Y = RF.Y[id];
							DT.a[a_size++] = make_pair(X, Y);
							constant &= X == X_0;
						}
						
						if (constant) {
							A.set_as(p, 2);
							continue;
						}
						
						A.set_as(p, 1);
						A.gen(p, p_tries, a_size, DT.a);
					}
				}
				
				if (f) Xid.erase(Xid.begin(), Xid.end());
			}

			bool split(bool dbg = 0) {
				delay = 0;
				if (leaf()) return 0;
				
				gen_spl(p_count);
				spl = A.find_best_spl();
				thres = spl.first, attr = spl.second;
				
				return attr != -1;
			}
			
			node* new_child() {
				node *u;
				if (DT.trash.empty()) {
					u = new node(RF, DT, dep + 1, A.cons);
				} else {
					u = DT.trash.back();
					DT.trash.pop_back();
					if (!u->Xid.empty()) u->Xid.clear();
					u->Xid = {};
					u->dep = dep + 1;
					u->A.reset();
					u->A.cons = A.cons;
					u->spl = {0.0, -1};
					u->thres = 0.0;
					u->attr = 0;
					u->delay = 1;
					u->old = 1;
					u->ls = u->rs = nullptr;
				}
				return u;
			}

			void separate() {
				ls = new_child();
				rs = new_child();
				thres = spl.first, attr = spl.second;
				
				// Proteção contra tamanhos inválidos
				if (Xid.size() > 1000000) {
					std::cerr << "ERRO CRÍTICO: Xid muito grande em separate(): " << Xid.size() << std::endl;
					throw std::length_error("Xid corrompido em separate()");
				}
				
				// Use capacidade limitada para reserve
				size_t safe_size = std::min(Xid.size(), static_cast<size_t>(100000));
				
				try {
					if (safe_size > 0) {
						ls->Xid.reserve(safe_size);
						rs->Xid.reserve(safe_size);
					}
				} catch (const std::exception& e) {
					std::cerr << "ERRO no reserve - Xid.size(): " << Xid.size() << ", safe_size: " << safe_size << std::endl;
					throw;
				}
				
				for (int id : Xid) {
					const auto &X = RF.X[id][attr];
					if (X <= thres) {
						ls->Xid.push_back(id);
					} else {
						rs->Xid.push_back(id);
					}
				}
				
				int &cnt = ls->A.n_1 = 0;
				for (int id : ls->Xid) if (RF.Y[id]) ++cnt;
				ls->A.n = ls->Xid.size();
				rs->A.n = rs->Xid.size();
				rs->A.n_1 = A.n_1 - cnt;
				Xid.clear();
				Xid.shrink_to_fit();
			}
			
			void build() {
				if (!split()) return;
				separate();
				ls->build();
				rs->build();
			}
			
			void destroy() {
				if (ls != nullptr) ls->destroy();
				if (rs != nullptr) rs->destroy();
				Xid.clear();
				free(ls);
				free(rs);
			}
			
			void concentrate() {
				if (ls == nullptr) return;
				
				A.n = static_cast<int>(Xid.size());
				if (A.n < 0) {
					A.n = 0;
				}
				
				A.n_1 = 0;
				if (A.n > 0) {
					for (int id : Xid) {
						if (id >= 0 && id < RF.Y.size() && RF.Y[id]) {
							A.n_1++;
						}
					}
				}
				
				if (A.n > 1000000) {
					std::cerr << "ERRO: A.n muito grande em concentrate(): " << A.n << std::endl;
					A.n = 0;
					return;
				}
				
				if (A.n > 0) {
					try {
						Xid.reserve(A.n);
					} catch (const std::exception& e) {
						std::cerr << "ERRO no reserve em concentrate() - A.n: " << A.n << std::endl;
					}
				}
				
				concentrate(Xid);
				ls = rs = nullptr;
			}
			void concentrate(vector<int> &Xids) {
				if (ls == nullptr) {
					if (Xids.empty()) Xids = std::move(Xid);
					else Xids.insert(Xids.end(), make_move_iterator(Xid.begin()), make_move_iterator(Xid.end()));
				} else {
					ls->concentrate(Xids);
					rs->concentrate(Xids);
					DT.trash.push_back(ls);
					DT.trash.push_back(rs);
				}
			}
			
			void collect() {
				if (ls == nullptr) return;
				
				if (A.n < 0) {
					A.n = static_cast<int>(Xid.size());
					if (A.n < 0) A.n = 0;
				}
				
				if (A.n > 1000000) {
					A.n = static_cast<int>(Xid.size());
					if (A.n > 1000000) A.n = 0;
				}
				
				if (A.n > 0) {
					try {
						Xid.reserve(A.n);
					} catch (const std::exception& e) {
						// Continue sem reservar em caso de erro
					}
				}
				
				collect(Xid);
			}
			void collect(vector<int> &Xids) {
				if (ls == nullptr) {
					Xids.insert(Xids.end(), Xid.begin(), Xid.end());
				} else {
					ls->collect(Xids);
					rs->collect(Xids);
				}
			}
			
			double qry(const vector<Tx> &X) {
				if (delay) {
					if (split()) separate();
				}
				if (ls == nullptr) {
					if (A.n <= 0) return 0.0;
					return (double) A.n_1 / A.n;
				}
				return (X[attr] <= thres) ? ls->qry(X) : rs->qry(X);
			}

			void add_leaf(int id) {
				if (ls == nullptr) Xid.push_back(id);
				else (RF.X[id][attr] <= thres) ? ls->add_leaf(id) : rs->add_leaf(id);
			}
			void add(int id) {
				const auto &X = RF.X[id];
				const auto &Y = RF.Y[id];
				if (A.add(X, Y)) gen_spl(p_count);
				if (delay) return;
				else if (ls == nullptr && !leaf()) {
					if (split()) separate();
				} else if (best_split_changed()) {
					concentrate();
					if (dly) delay = 2;
					else build();
				} else if (ls != nullptr) {
					(RF.X[id][attr] <= thres) ? ls->add(id) : rs->add(id);
				}
			}

			void del_leaf(int id) {
				if (ls == nullptr) erase(Xid, id);
				else (RF.X[id][attr] <= thres) ? ls->del_leaf(id) : rs->del_leaf(id);
			}
			void del(int id) {
				const auto &X = RF.X[id];
				const auto &Y = RF.Y[id];
				if (A.del(X, Y)) gen_spl(p_count);
				if (delay) return;
				else if (ls != nullptr && leaf()) {
					concentrate();
				} else if (best_split_changed()) {
					concentrate();
					if (dly) delay = 2;
				} else if (ls != nullptr) {
					(RF.X[id][attr] <= thres) ? ls->del(id) : rs->del(id);
				}
			}

			void del_leaf(const vector<int> &ids) {
				if (ids.size() <= 5) {
					for (int id : ids) del_leaf(id);
					return;
				}
				
				if (ls == nullptr) {
					int del_cnt = 0, n = Xid.size();
					unordered_map<int, bool> to_del;
					for (int id : ids) to_del[id] = 1;
					for (int i = 0; i + del_cnt < n; ++i) {
						int id = Xid[i];
						if (to_del.count(id)) {
							del_cnt += 1;
							swap(Xid[i], Xid[n - del_cnt]);
							--i;
						}
						if (del_cnt == ids.size()) break;
					}
					for (int i = 0; i < del_cnt; ++i) Xid.pop_back();
				} else {
					vector<int> ids_l, ids_r;
					ids_l.reserve(ids.size());
					ids_r.reserve(ids.size());
					for (int id : ids) {
						const auto &X = RF.X[id][attr];
						if (X <= thres) {
							ids_l.push_back(id);
						} else {
							ids_r.push_back(id);
						}
					}
					if (!ids_l.empty()) ls->del_leaf(ids_l);
					if (!ids_r.empty()) rs->del_leaf(ids_r);
				}
			}
			void del(const vector<int> &ids) {
				int n = ids.size();
				if (n <= 5) {
					for (int id : ids) del(id);
					return;
				}
				vector<const vector<Tx>*> Xs;
				vector<Ty> Ys;
				Xs.reserve(n), Ys.reserve(n);
				for (int id : ids) {
					Xs.emplace_back(&RF.X[id]);
					Ys.emplace_back(RF.Y[id]);
				}
				if (n * 3 > A.n) {
					concentrate();
					if (dly) delay = 1;
					else build();
					return;
				}
				if (A.del(Xs, Ys)) gen_spl(p_count);
				
				if (delay) return;
				else if (ls != nullptr && leaf()) {
					concentrate();
				} else if (best_split_changed()) {
					concentrate();
					if (dly) delay = 1;
					else build();
				} else if (ls != nullptr) {
					vector<int> ids_l, ids_r;
					ids_l.reserve(n);
					ids_r.reserve(n);
					for (int id : ids) {
						const auto &X = RF.X[id][attr];
						if (X <= thres) {
							ids_l.push_back(id);
						} else {
							ids_r.push_back(id);
						}
					}
					if (!ids_l.empty()) ls->del(ids_l);
					if (!ids_r.empty()) rs->del(ids_r);
				}
			}
			
			bool best_split_changed() {
				pair<Tx, int> best = A.find_best_spl();
				bool f = this->spl != best;
				this->spl = best;
				return f;
			}
			
			bool leaf() {
				return dep >= DT.max_dep || A.n < DT.min_split_size || A.n_1 == 0 || A.n_1 == A.n;
			}
		};
		node *root;

		void serialize(std::ostream &out) const {
			out.write((char*)&max_dep, sizeof(max_dep));
			out.write((char*)&min_split_size, sizeof(min_split_size));
			
			if (root) {
				root->serialize(out);
			}
		}

		static decision_tree* deserialize(std::istream &in, const random_forest &RF) {
			int max_dep_local, min_split_size_local;
			in.read((char*)&max_dep_local, sizeof(max_dep_local));
			in.read((char*)&min_split_size_local, sizeof(min_split_size_local));
			vector<int> fake_vec;
			decision_tree* t = new decision_tree(RF, fake_vec, max_dep_local, min_split_size_local);
			t->root = node::deserialize(in, *t, RF);
			return t;
		}

		// Função para imprimir a estrutura da árvore
		void print_tree(const std::string &prefix = "") const {
			if (root) {
				cout << prefix << "Decision Tree (max_dep=" << max_dep << ", min_split=" << min_split_size << "):" << endl;
				print_node_compact(root, prefix + "  ", 0);
			} else {
				cout << prefix << "Empty tree" << endl;
			}
		}

		void print_node_compact(node* n, const std::string &prefix, int depth, int max_depth = 4) const {
			if (!n || depth > max_depth) {
				if (depth > max_depth) cout << prefix << "... (truncated)" << endl;
				return;
			}
			
			string node_type = n->leaf() ? "LEAF" : "NODE";
			cout << prefix << node_type << " [d=" << n->dep << ", n=" << n->A.n << ", n1=" << n->A.n_1;
			
			if (!n->leaf()) {
				cout << "] split: attr=" << n->attr << ", thres=" << std::fixed << std::setprecision(3) << n->thres;
				cout << " (parent nodes have n=0 by design after split)";
			} else {
				double prob = (n->A.n > 0) ? (double)n->A.n_1 / n->A.n : 0.0;
				cout << "] pred=" << std::fixed << std::setprecision(3) << prob;
			}
			cout << endl;
			
			if (n->ls || n->rs) {
				if (n->ls) {
					cout << prefix << "├─L: ";
					print_node_compact(n->ls, prefix + "│    ", depth + 1, max_depth);
				}
				if (n->rs) {
					cout << prefix << "└─R: ";
					print_node_compact(n->rs, prefix + "     ", depth + 1, max_depth);
				}
			}
		}

		void print_node(node* n, const std::string &prefix, bool is_root = false) const {
			if (!n) return;
			
			string node_type = n->leaf() ? "LEAF" : "NODE";
			cout << prefix << node_type << " [dep=" << n->dep << ", samples=" << n->A.n 
				 << ", class1=" << n->A.n_1 << ", delay=" << n->delay << "]";
			
			if (!n->leaf()) {
				cout << " split: attr=" << n->attr << ", thres=" << n->thres;
			} else {
				double prob = (n->A.n > 0) ? (double)n->A.n_1 / n->A.n : 0.0;
				cout << " prediction=" << prob;
			}
			cout << endl;
			
			if (n->ls || n->rs) {
				if (n->ls) {
					cout << prefix << "├─L: ";
					print_node(n->ls, prefix + "│    ");
				}
				if (n->rs) {
					cout << prefix << "└─R: ";
					print_node(n->rs, prefix + "     ");
				}
			}
		}
		
//		decision_tree(): RF(unusable_forest) {}
		decision_tree(const random_forest &RF, vector<int> &Xid, int max_dep, int min_split_size):
			RF(RF), max_dep(max_dep), min_split_size(min_split_size), a(Xid.size()) {
			root = new node(RF, *this, 0);
			root->Xid = std::move(Xid);
			
			trash.reserve(100000);
			
			#pragma omp critical
			{
				this->mt = mt19937(rd());
			}

			int &cnt = root->A.n_1;
			for (int id : root->Xid) if (RF.Y[id]) ++cnt;
			root->A.n = root->Xid.size();
			root->build();
		}
		
		double qry(const vector<Tx> &X) {
			return root->qry(X);
		}
		
		void add(int id) {
			root->add_leaf(id);
			root->add(id);
		}
		
		void del(int id) {
			root->del_leaf(id);
			root->del(id);
		}
		
		void del(const vector<int> &ids) {
			root->del_leaf(ids);
			root->del(ids);
		}
		// void del(node *u, const vector<int> &ids) {
		// 	if (ids.size() <= 3) {
		// 		for (int id : ids) u->del(id);
		// 		return;
		// 	}
		// 	u->del(ids);
		// 	if (u->delay) return;
		// 	else if (u->leaf()) {
		// 		// u->destroy_children();
		// 	}
		// 	else if (u->Xid.size() > 1000 && ids.size() * 3 >= u->Xid.size()) {
		// 		// u->destroy_children();
		// 		if (dly) u->delay = 1;
		// 		else u->separate(), u->ls->build(), u->rs->build();
		// 	}
		// 	else if (u->best_split_changed()) {
		// 	//	cerr << "tagged " << u->Xid.size() << endl;
		// 		// u->destroy_children();
		// 		if (dly) u->delay = 2;
		// 		else u->separate(), u->ls->build(), u->rs->build();
		// 	}
		// 	else if (u->ls != nullptr) {
		// 		vector<int> ids_l, ids_r;
		// 		for (int id : ids) RF.X[id][u->attr] <= u->thres ? ids_l.push_back(id) : ids_r.push_back(id);
		// 		// cerr << ids.size() << " -> " << ids_l.size() << ' ' << ids_r.size() << endl;
		// 		if (ids_l.size()) del(u->ls, ids_l);
		// 		if (ids_r.size()) del(u->rs, ids_r);
		// 	}
		// }

		
		void develop() { develop(root); };
		void develop(node *u) {
			if (u->delay) {
				// counter += u->Xid.size();
				if (u->delay == 1) {
					u->build();
				} else {
					u->separate();
					u->ls->build();
					u->rs->build();
				}
				return;
			}
			if (u->ls != nullptr) develop(u->ls);
			if (u->rs != nullptr) develop(u->rs);
		}

		void clean_up() {
			for (auto u : trash) delete u;
			trash.clear();
		}

		vector<node*> trash;
	};
	decision_tree **tr;
	//vector<decision_tree> tr;

	void serialize(const std::string &fname) const {
		std::ofstream out(fname, std::ios::binary);
		out.write((char*)&d, sizeof(d));
		out.write((char*)&n, sizeof(n));
		out.write((char*)&C, sizeof(C));
		out.write((char*)&T, sizeof(T));
		out.write((char*)&k, sizeof(k));

		// Serializar X_binary
		for (int i = 0; i < d; ++i) {
			bool val = X_binary[i];
			out.write((char*)&val, sizeof(val));
		}

		for (int i = 0; i < n; ++i) {
			if (X[i].empty()) {
				for (int j = 0; j < d; ++j) {
					double zero_val = 0.0;
					out.write((char*)&zero_val, sizeof(zero_val));
				}
			} else {
				for (int j = 0; j < d; ++j) {
					out.write((char*)&X[i][j], sizeof(X[i][j]));
				}
			}
		}

		// Serializar labels Y
		for (int i = 0; i < n; ++i) {
			out.write((char*)&Y[i], sizeof(Y[i]));
		}

		// Serializar associações at
		for (int i = 0; i < n; ++i) {
			uint32_t at_size = static_cast<uint32_t>(at[i].size());
			out.write((char*)&at_size, sizeof(at_size));
			if (at_size > 0) {
				out.write((char*)at[i].data(), at_size * sizeof(int));
			}
		}

		// Serializar cada árvore
		for (int i = 0; i < T; ++i) {
			tr[i]->serialize(out);
		}
		out.close();
	}
	static random_forest* deserialize(const std::string &fname) {
		std::ifstream in(fname, std::ios::binary);
		if (!in.is_open()) {
			cerr << "ERROR: Cannot open file " << fname << " for reading" << endl;
			return nullptr;
		}
		
		int d_local, n_local, C_local, T_local, k_local;
		in.read((char*)&d_local, sizeof(d_local));
		in.read((char*)&n_local, sizeof(n_local));
		in.read((char*)&C_local, sizeof(C_local));
		in.read((char*)&T_local, sizeof(T_local));
		in.read((char*)&k_local, sizeof(k_local));
		
		// Verificar se os valores lidos são razoáveis
		if (d_local <= 0 || n_local <= 0 || T_local <= 0 || k_local <= 0) {
			cerr << "ERROR: Invalid parameters read from file: d=" << d_local 
			     << ", n=" << n_local << ", T=" << T_local << ", k=" << k_local << endl;
			return nullptr;
		}

		// Deserializar X_binary
		vector<bool> X_binary_local(d_local);
		for (int i = 0; i < d_local; ++i) {
			bool val;
			in.read((char*)&val, sizeof(val));
			X_binary_local[i] = val;
		}

		vector<vector<double>> X_local(n_local, vector<double>(d_local));
		for (int i = 0; i < n_local; ++i) {
			for (int j = 0; j < d_local; ++j) {
				in.read((char*)&X_local[i][j], sizeof(X_local[i][j]));
			}
		}

		// Deserializar labels Y
		vector<int> Y_local(n_local);
		for (int i = 0; i < n_local; ++i) {
			in.read((char*)&Y_local[i], sizeof(Y_local[i]));
		}

		random_forest* rf = new random_forest();
		rf->d = d_local;
		rf->n = n_local;
		rf->C = C_local;
		rf->T = T_local;
		rf->k = k_local;
		rf->X = X_local;
		rf->Y = Y_local;
		rf->X_binary = X_binary_local;

		// Inicializar variáveis globais necessárias
		id_T.resize(T_local);
		for (int i = 0; i < T_local; ++i) id_T[i] = i;
		
		id_d.resize(d_local);
		for (int i = 0; i < d_local; ++i) id_d[i] = i;

		// Deserializar associações at
		rf->at.resize(n_local);
		for (int i = 0; i < n_local; ++i) {
			uint32_t at_size;
			in.read((char*)&at_size, sizeof(at_size));
			// Adicionar validação para evitar valores inválidos
			if (at_size > 0 && at_size <= T_local) {  // at_size não pode ser maior que T
				rf->at[i].resize(at_size);
				in.read((char*)rf->at[i].data(), at_size * sizeof(int));
			} else if (at_size > T_local) {
				cerr << "ERROR: Invalid at_size " << at_size << " for sample " << i 
				     << " (max expected: " << T_local << ")" << endl;
				rf->at[i].clear();  // Deixar vazio em caso de erro
			}
		}

		rf->tr = new decision_tree*[T_local];
		for (int i = 0; i < T_local; ++i) {
			rf->tr[i] = decision_tree::deserialize(in, *rf);
		}
		in.close();
		return rf;
	}
	
	random_forest() {}
	random_forest(vector<vector<Tx>> X, vector<Ty> Y, int T = 100, int k = 10, int max_dep = 15, int min_split_size = 10):
		X(X), Y(Y), T(T), k(k) {
		n = X.size(), d = X[0].size();
		C = *max_element(Y.begin(), Y.end()) + 1;
		X_binary = vector<bool>(d, 0);
	
		id_d.resize(d);
		
		unordered_map<Tx, bool> to;
		for (int p = 0; p < d; ++p) {
			to.clear();
			for (int i = 0; i < n; ++i) {
				Tx cur = X[i][p];
				if (!to.count(cur)) to[cur] = to.size();
				if (to.size() > 2) break;
			}
			if (to.size() == 2) X_binary[p] = 1;
		}
		
		vector<vector<int>> Xids;
		data_distribution(Xids);
		for (int i = 0; i < d; ++i) id_d[i] = i;
		at.resize(n);
		tr = new decision_tree*[T];

		for (int t = 0; t < T; ++t) {
			for (int i : Xids[t]) at[i].push_back(t);
		}
		
		#pragma omp parallel for schedule(static)
		for (int t = 0; t < T; ++t) {
			tr[t] = new decision_tree(*this, Xids[t], max_dep, min_split_size);
		}
	}
	
	void data_distribution(vector<vector<int>> &Xids) {
		id_T.resize(T);
		for (int i = 0; i < T; ++i) id_T[i] = i;
		
		Xids.resize(T);
		for (int i = 0; i < n; ++i) {
			shuffle(id_T.begin(), id_T.end(), mt);
			for (int j = 0; j < k; ++j) {
				int t = id_T[j];
				Xids[t].push_back(i);
			}
		}
	}

	void qry(const vector<Tx> &X, vector<double> &res) {
		// cerr << "qry start" << endl;
		res.resize(C);
		for (int c = 0; c < C; ++c) res[c] = 0;
		double &r_0 = res[0], &r_1 = res[1];

		#pragma omp parallel for schedule(dynamic)
		for (int t = 0; t < T; ++t) {
			double p = tr[t]->qry(X);
			#pragma omp atomic
			r_0 += 1 - p;
			#pragma omp atomic
			r_1 += p;
		}
		
		for (int c = 0; c < C; ++c) res[c] /= T;
		
		// cerr << "qry end" << endl;
	}

	void add(const vector<Tx> &X, const Ty &Y) {
		int id = this->X.size();
		this->X.push_back(X);
		this->Y.push_back(Y);
		
		shuffle(id_T.begin(), id_T.end(), mt);
		at.push_back({});

		for (int j = 0; j < k; ++j) {
			int t = id_T[j];
			at[id].push_back(t);
		}
		
		#pragma omp parallel for schedule(static)
		for (int j = 0; j < k; ++j) {
			int t = id_T[j];
			tr[t]->add(id);
		}
	}
	
	void del(const vector<Tx> &X, const Ty &Y) {
		for (int i = 0; i < X.size(); ++i) {
			if (this->X[i] == X && this->Y[i] == Y) del(i);
		}
	}
	void del(int id) {
		const vector<int> &tid = at[id];
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < tid.size(); ++i) {
			tr[tid[i]]->del(id);
		}
		X[id] = vector<Tx>(0), Y[id] = 0;
	}
	
	void del(const vector<int> &ids, bool clean_tags = 0) {
		vector<vector<int>> del_ids(T);
		for (int id : ids) {
			for (int tid : at[id]) {
				del_ids[tid].push_back(id);
			}
			at[id].clear();
		}
		for (int t = 0; t < T; ++t) {
			if (del_ids[t].size()) tr[t]->del(del_ids[t]);
		}
		
		if (clean_tags) develop();
	}

	void develop() {
		for (int t = 0; t < T; ++t) tr[t]->develop();
	}
	
	void clean_up(bool force = 0) {
		long long sum = 0;
		#pragma omp parallel for schedule(dynamic)
		for (int t = 0; t < T; ++t) {
			#pragma omp atomic
			sum += tr[t]->trash.size();
		}
		if (!force && sum < 1000000) return;
		#pragma omp parallel for schedule(dynamic) 
		for (int t = 0; t < T; ++t) {
			tr[t]->clean_up();
		}
	}

	// Função para imprimir informações da floresta
	void print_forest_info() const {
		cout << "=== Random Forest Info ===" << endl;
		cout << "Trees: " << T << ", Features: " << d << ", Samples: " << n 
			 << ", Classes: " << C << ", k: " << k << endl;
		cout << "==========================" << endl;
	}

	// Função para imprimir árvores específicas (útil para comparação)
	void print_trees(const vector<int> &tree_indices) const {
		for (int idx : tree_indices) {
			if (idx >= 0 && idx < T) {
				cout << "\n--- Tree " << idx << " ---" << endl;
				tr[idx]->print_tree();
			}
		}
	}

	// Função para imprimir algumas árvores (primeiras n) de forma compacta
	void print_first_trees(int n_trees = 3) const {
		for (int i = 0; i < min(n_trees, T); ++i) {
			cout << "\n--- Tree " << i << " (compact) ---" << endl;
			if (tr[i]->root) {
				tr[i]->print_node_compact(tr[i]->root, "  ", 0, 3); // Max depth 3 para compacidade
			} else {
				cout << "  Empty tree" << endl;
			}
		}
	}

	// Função para comparar duas florestas imprimindo as primeiras árvores
	static void compare_forests(const random_forest &rf1, const random_forest &rf2, int n_trees = 2) {
		cout << "\n========== FOREST COMPARISON ==========" << endl;
		cout << "\nORIGINAL FOREST:" << endl;
		rf1.print_forest_info();
		rf1.print_first_trees(n_trees);
		
		cout << "\nDESERIALIZED FOREST:" << endl;
		rf2.print_forest_info();
		rf2.print_first_trees(n_trees);
		cout << "\n=======================================" << endl;
	}

	// Função para verificar estruturalmente se duas árvores são idênticas
	static bool trees_structurally_equal(decision_tree::node* n1, decision_tree::node* n2, int depth = 0) {
		// Ambos são nulos
		if (!n1 && !n2) return true;
		
		// Um é nulo, outro não
		if (!n1 || !n2) return false;
		
		// Verificar se ambos são folhas ou ambos são nós internos
		bool leaf1 = n1->leaf(), leaf2 = n2->leaf();
		if (leaf1 != leaf2) return false;
		
		if (leaf1) {
			// Para folhas, comparar apenas se têm dados válidos ou não
			bool has_data1 = (n1->A.n > 0);
			bool has_data2 = (n2->A.n > 0);
			if (has_data1 && has_data2) {
				// Ambas têm dados - comparar predição
				double pred1 = (double)n1->A.n_1 / n1->A.n;
				double pred2 = (double)n2->A.n_1 / n2->A.n;
				return abs(pred1 - pred2) < 1e-6;
			}
			return has_data1 == has_data2; // Ambas vazias ou ambas com dados
		} else {
			// Para nós internos, comparar split
			if (abs(n1->attr - n2->attr) > 1e-9) return false;
			if (abs(n1->thres - n2->thres) > 1e-6) return false;
			
			// Recursão nos filhos
			return trees_structurally_equal(n1->ls, n2->ls, depth + 1) &&
				   trees_structurally_equal(n1->rs, n2->rs, depth + 1);
		}
	}

	// Função melhorada para comparação visual das florestas
	static void verify_forests_integrity(const random_forest &rf1, const random_forest &rf2, int n_trees = 5) {
		cout << "\n========== FOREST INTEGRITY CHECK ==========" << endl;
		
		// Comparar metadados básicos
		cout << "Basic Metadata:" << endl;
		cout << "   Trees (tamanho): " << rf1.T << " vs " << rf2.T;
		if (rf1.T == rf2.T) cout << " OK"; else cout << " FAIL";
		cout << endl;
		
		cout << "   Features (tamanho): " << rf1.X[0].size() << " vs " << rf2.X[0].size();
		if (rf1.X[0].size() == rf2.X[0].size()) cout << " OK"; else cout << " FAIL";
		cout << endl;
		
		cout << "   Samples (tamanho): " << rf1.X.size() << " vs " << rf2.X.size();
		if (rf1.X.size() == rf2.X.size()) cout << " OK"; else cout << " FAIL";
		cout << endl;
		
		cout << "   k parameter: " << rf1.k << " vs " << rf2.k;
		if (rf1.k == rf2.k) cout << " OK"; else cout << " FAIL";
		cout << endl;
		
		// Verificar se as estruturas das primeiras árvores são idênticas
		cout << "\nTree Structure Verification:" << endl;
		int trees_to_check = min(n_trees, (int)rf1.T);
		int identical_trees = 0;
		
		for (int i = 0; i < trees_to_check; i++) {
			bool is_identical = trees_structurally_equal(rf1.tr[i]->root, rf2.tr[i]->root);
			cout << "   Tree " << i << ": ";
			if (is_identical) {
				cout << "IDENTICAL";
				identical_trees++;
			} else {
				cout << "DIFFERENT";
			}
			cout << endl;
		}
		
		cout << "   Summary: " << identical_trees << "/" << trees_to_check << " trees are identical" << endl;
		
		// Teste de consistência de predições
		cout << "\nPrediction Consistency Test:" << endl;
		int test_samples = min(5, (int)rf1.X.size());
		int consistent_predictions = 0;
		
		for (int i = 0; i < test_samples; i++) {
			double pred1 = 0.0, pred2 = 0.0;
			
			// Calcular predição para rf1
			for (int t = 0; t < rf1.T; t++) {
				pred1 += rf1.tr[t]->qry(rf1.X[i]);
			}
			pred1 /= rf1.T;
			
			// Calcular predição para rf2
			for (int t = 0; t < rf2.T; t++) {
				pred2 += rf2.tr[t]->qry(rf2.X[i]);
			}
			pred2 /= rf2.T;
			
			bool consistent = abs(pred1 - pred2) < 1e-6;
			if (consistent) consistent_predictions++;
			
			cout << "   Sample " << i << ": " << fixed << setprecision(6) 
				 << pred1 << " vs " << pred2;
			if (consistent) cout << " OK"; else cout << " FAIL";
			cout << endl;
		}
		
		cout << "   Summary: " << consistent_predictions << "/" << test_samples << " predictions are consistent" << endl;
		cout << "\n===============================================" << endl;
	}
};