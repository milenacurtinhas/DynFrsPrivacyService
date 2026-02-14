/**
 * DynFrs Python Binding - Wrapper pybind11 para DynFrs.h
 * Expõe funcionalidades C++ para Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../DynFrs.h"

namespace py = pybind11;

/**
 * Wrapper class para random_forest
 * Expõe funcionalidades essenciais para Python
 */
class RandomForestWrapper {
private:
    random_forest* rf;
    
public:
    // Construtor
    RandomForestWrapper(
        std::vector<std::vector<double>> X,
        std::vector<int> Y,
        int T = 100,
        int k = 10,
        int max_dep = 15,
        int min_split_size = 10
    ) {
        rf = new random_forest(X, Y, T, k, max_dep, min_split_size);
    }
    
    // Destrutor
    ~RandomForestWrapper() {
        if (rf != nullptr) {
            delete rf;
        }
    }
    
    // Query (predição)
    std::vector<double> query(const std::vector<double>& X) {
        std::vector<double> result;
        rf->qry(X, result);
        return result;
    }
    
    // Adicionar amostra
    void add_sample(const std::vector<double>& X, int Y) {
        rf->add(X, Y);
    }
    
    // Remover amostra (unlearning)
    void delete_sample(int id) {
        rf->del(id);
    }
    
    // Remover múltiplas amostras
    void delete_samples(const std::vector<int>& ids) {
        rf->del(ids, false);
    }
    
    // Serializar modelo
    void serialize(const std::string& filename) {
        rf->serialize(filename);
    }
    
    // Deserializar modelo (estático)
    static RandomForestWrapper* deserialize(const std::string& filename) {
        random_forest* loaded_rf = random_forest::deserialize(filename);
        if (loaded_rf == nullptr) {
            throw std::runtime_error("Failed to deserialize model from: " + filename);
        }
        
        // Criar wrapper
        RandomForestWrapper* wrapper = new RandomForestWrapper({}, {}, 1, 1, 1, 1);
        delete wrapper->rf;  // Deletar RF vazio
        wrapper->rf = loaded_rf;  // Usar RF carregado
        
        return wrapper;
    }
    
    // Desenvolver árvores delayed
    void develop() {
        rf->develop();
    }
    
    // Limpar memória
    void clean_up() {
        rf->clean_up(true);
    }
    
    // Obter informações
    int get_num_trees() const {
        return rf->T;
    }
    
    int get_num_samples() const {
        return rf->n;
    }
    
    int get_num_features() const {
        return rf->d;
    }
    
    int get_num_classes() const {
        return rf->C;
    }
    
    // Imprimir informações (debug)
    void print_info() const {
        rf->print_forest_info();
    }
};


// Módulo pybind11
PYBIND11_MODULE(dynfrs_cpp, m) {
    m.doc() = "DynFrs C++ binding - Dynamic Random Forest with Machine Unlearning";
    
    py::class_<RandomForestWrapper>(m, "RandomForest")
        .def(py::init<
            std::vector<std::vector<double>>,
            std::vector<int>,
            int, int, int, int
        >(),
        py::arg("X"),
        py::arg("Y"),
        py::arg("T") = 100,
        py::arg("k") = 10,
        py::arg("max_dep") = 15,
        py::arg("min_split_size") = 10,
        "Construct a Random Forest classifier"
        )
        .def("query", &RandomForestWrapper::query,
             py::arg("X"),
             "Query the forest for prediction probabilities")
        .def("add_sample", &RandomForestWrapper::add_sample,
             py::arg("X"), py::arg("Y"),
             "Add a new sample to the forest")
        .def("delete_sample", &RandomForestWrapper::delete_sample,
             py::arg("id"),
             "Delete a sample from the forest (machine unlearning)")
        .def("delete_samples", &RandomForestWrapper::delete_samples,
             py::arg("ids"),
             "Delete multiple samples from the forest")
        .def("serialize", &RandomForestWrapper::serialize,
             py::arg("filename"),
             "Serialize the forest to a binary file")
        .def_static("deserialize", &RandomForestWrapper::deserialize,
             py::arg("filename"),
             "Deserialize a forest from a binary file")
        .def("develop", &RandomForestWrapper::develop,
             "Develop delayed trees")
        .def("clean_up", &RandomForestWrapper::clean_up,
             "Clean up memory")
        .def("get_num_trees", &RandomForestWrapper::get_num_trees,
             "Get number of trees")
        .def("get_num_samples", &RandomForestWrapper::get_num_samples,
             "Get number of samples")
        .def("get_num_features", &RandomForestWrapper::get_num_features,
             "Get number of features")
        .def("get_num_classes", &RandomForestWrapper::get_num_classes,
             "Get number of classes")
        .def("print_info", &RandomForestWrapper::print_info,
             "Print forest information");
}
