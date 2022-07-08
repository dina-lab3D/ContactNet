#include <fstream>
#include <Molecule.h>
#include <Atom.h>
#include "cnpy.h"

std::string trimExtension(const std::string file_name) {
  if(file_name[file_name.size()-4] == '.')
    return file_name.substr(0, file_name.size() - 4);
  return file_name;
}

void computeDistMat(const Molecule<Atom> &mol1, std::vector<float> &output_distances, float thr = 16.0)
{
  for (unsigned int mol1Index = 0; mol1Index < mol1.size(); mol1Index++)
    {
      // create a distance matrix
      for (unsigned int mol2Index = 0; mol2Index < mol1.size(); mol2Index++)
        {
          float d = mol1(mol1Index).dist(mol1(mol2Index));
          if (d > thr) { // to far
            output_distances.push_back(0.0);
          } else {
            if(d < 0.0001) output_distances.push_back(1.0);
            else output_distances.push_back(1.0/d); //normalize the dist
          }
        }
    }
}

void saveMat(const std::vector<float>& mat, unsigned long n, std::string file_name)
{
  std::cout << "saving distogram: " << file_name << std::endl;
  cnpy::npy_save(file_name, &mat[0], {n, n}, "w");
}

int createDistogram(std::string pdb_file_name)
{
  std::ifstream pdb_file(pdb_file_name);
  if (!pdb_file) {
    std::cerr << "PDB file not found " << pdb_file_name << std::endl;
    return 1;
  }
  Molecule<Atom> mol;
  mol.readPDBfile(pdb_file_name, PDB::CAlphaSelector());
  std::vector<float> distogram;
  computeDistMat(mol, distogram);
  std::string file_name = trimExtension(pdb_file_name) + ".npy";
  saveMat(distogram, mol.size(), file_name);
  return 0;
}

int main(int argc, char **argv) {
  // output arguments
  for (int i = 0; i < argc; i++) std::cerr << argv[i] << " ";
  std::cerr << std::endl;

  if(argc == 1) {
    std::cout << "Usage: " << argv[0] << "<pdb1> <pdb2> ..." << std::endl;
    return 0;
  }

  for(int i = 1; i < argc; i++) {
    createDistogram(argv[i]);
  }
  return 0;
}
