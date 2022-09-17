#include "RigidTrans3.h"
#include "Molecule.h"
#include "Atom.h"
#include "cnpy.h"

#include <boost/algorithm/string.hpp>
#include <filesystem>
namespace fs = std::filesystem;

#include <fstream>
#include <iostream>

std::string trimExtension(const std::string file_name) {
  if(file_name[file_name.size()-4] == '.')
    return file_name.substr(0, file_name.size() - 4);
  return file_name;
}

void saveMat(unsigned long n, unsigned long m, std::vector<float>& mat, std::string file_name) {
  std::cout << "saving distogram: " << file_name<<std::endl;
  cnpy::npy_save(file_name, &mat[0], {n, m}, "w");
}

void readTransFile(std::string filename, std::vector<RigidTrans3>& trans,std::vector<int>& trans_index, unsigned int transNum = 0) {
  std::ifstream inS(filename);
  if (!inS) {
    std::cerr << "Problem opening transformation file " << filename << std::endl;
    return ;
  }
  std::string line;
  unsigned int counter = 0;
  int transNumber = 0;
  while (!inS.eof()) {
    getline(inS, line);
    boost::trim(line);  // remove all spaces
    // skip comments
    if (line[0] == '#' || line[0] == '\0') continue;
    std::vector<std::string> split_results;
    boost::split(split_results, line, boost::is_any_of(" :|\t"),
                 boost::token_compress_on);
    //std::cerr << split_results.size() << std::endl;
    if (split_results.size() < 7) continue;
    transNumber = 0;
    try {
      transNumber = std::stoi(split_results[0]);
      trans_index.push_back(transNumber);
    }
    catch (...) {
      std::cout << "Invalid input. Please try again!\n";
      continue;
    }


    // read the last 6 numbers from any file format
    unsigned int trans_index = split_results.size() - 6;
    RigidTrans3 tr(Vector3(std::stof(split_results[trans_index].c_str()),
                           std::stof(split_results[trans_index+1].c_str()),
                           std::stof(split_results[trans_index+2].c_str())),
                   Vector3(std::stof(split_results[trans_index+3].c_str()),
                           std::stof(split_results[trans_index+4].c_str()),
                           std::stof(split_results[trans_index+5].c_str())));

    if(transNum > 0 && counter < transNum) {
      trans.push_back(tr);
    } else
      break;
    counter++;

    // std::cerr << tr << std::endl;
  }
  std::cerr << counter << " transforms were read from file " << filename << std::endl;
  inS.close();
}

void computeDistMat(const Molecule<Atom> &mol1, std::vector<float> &out, int thr = 16)
{
    for (unsigned int mol1Index = 0; mol1Index < mol1.size(); mol1Index++)
    {// create a distance matrix
        for (unsigned int mol2Index = 0; mol2Index < mol1.size(); mol2Index++)
        {



            float d = mol1(mol1Index).dist(mol1(mol2Index));
            if (d > 16)
            {//todo cutoff to far
                out.push_back(0);
            }
            else
            {
                if( d<0.0001)out.push_back(1);
                else out.push_back(1 / (d));//normlize the dist
//                  out.push_back(d);
            }
        }

    }
}

int computeDistMatrix(const Molecule<Atom> &mol1, const Molecule<Atom> &mol2,
                      std::vector<float>& output_distances, float thr = 16.0) {
  bool is_contact = false;

  // compute a distance matrix
  for(unsigned int mol1Index = 0; mol1Index < mol1.size() ; mol1Index++) {
    for(unsigned int mol2Index = 0; mol2Index < mol2.size() ; mol2Index++) {
      float d = mol1(mol1Index).dist(mol2(mol2Index));
      if (d > thr){
        output_distances.push_back(0);
      } else {
        if (!is_contact) { is_contact = true;}
//        if(d < 0.0001) output_distances.push_back(1.0);
        output_distances.push_back(1/d); //normalize the dist
      }
    }
  }
  if (!is_contact) { return 1; }
  return 0;
}


int main(int argc, char **argv) {

  for (int i = 0; i < argc; i++) std::cerr << argv[i] << " ";
  std::cerr << std::endl;

  if(argc == 1) {
    std::cout << "Usage: " << argv[0] << "<pdb1> <pdb2> [trans_file] [trans_num]" << std::endl;
    return 0;
  }
  if(argc > 3 && argc < 6) {
    // read the pdbs
    std::string pdb_file_name1(argv[1]), pdb_file_name2(argv[2]);
    std::ifstream pdb_file1(pdb_file_name1);
    if (!pdb_file1) {
      std::cerr << "PDB file not found " << pdb_file_name1 << std::endl;
      return 1;
    }
    std::ifstream pdb_file2(pdb_file_name2);
    if (!pdb_file2) {
      std::cerr << "PDB file not found " << pdb_file_name2 << std::endl;
      return 1;
    }
    Molecule<Atom> mol1, mol2;
    mol1.readPDBfile(pdb_file_name1, PDB::CAlphaSelector());
    mol2.readPDBfile(pdb_file_name2, PDB::CAlphaSelector());

    if(mol1.size()==0) {
      std::cerr << "No CA atoms " << pdb_file_name1 << std::endl;
      return 1;
    }
    if(mol2.size()==0) {
      std::cerr << "No CA atoms " << pdb_file_name2 << std::endl;
      return 1;
    }
    std::cout<< "Mol1 CA size: " << mol1.size()<<std::endl;
    std::cout<< "Mol2 CA size: " << mol2.size()<<std::endl;

    // save self distograms
    std::vector<float> distogram1, distogram2;
    computeDistMat(mol1, distogram1);
    computeDistMat(mol2, distogram2);
    std::string file_name1 = trimExtension(pdb_file_name1) + "_self_distogram.npy";
    std::string file_name2 = trimExtension(pdb_file_name2) + "_self_distogram.npy";
    saveMat(mol1.size(), mol1.size(), distogram1, file_name1);
    saveMat(mol2.size(), mol2.size(), distogram2, file_name2);


    // read transformations
//    std::vector<RigidTrans3> trans;
//    std::vector<int> trans_indices;
//
////    if(argc == 3) { // no transformations given, use identity
////      trans.push_back(RigidTrans3());
////    }
////    if(argc >=4) { // read transformations from a file
////      int transNum = 0;
//     if(argc < 5) {  std::cout<<"not engouthe  args "<<std::endl; }
//     int transNum = std::stoi(argv[4]);
//     readTransFile(argv[3], trans, trans_indices,transNum);
////    }
//
//    // calculate matrices
//    std::ofstream filenames("distograms.txt");
//    std::ofstream transfile("trans.txt");
//    for(unsigned int i=0; i<trans.size(); i++) {
//      Molecule<Atom> tmol2 = mol2;
//      tmol2.rigidTrans(trans[i]);
//      std::vector<float> mat;
//      computeDistMatrix(tmol2,mol1, mat);
//      std::string out_file = std::string(fs::path(pdb_file_name2).stem())+".pdbX"+
//      std::string(fs::path(pdb_file_name1).stem()) + ".pdbtransform_number_" + std::to_string(trans_indices[i]) ;
//      std::cout << i+1 << " " << trans[i] << " ";
//      saveMat( tmol2.size(),mol1.size(), mat, out_file);
//      filenames << out_file << std::endl;
//      transfile << i+1 << "\t" << trans[i] << std::endl;
//    }
  }
  return 1;
}
