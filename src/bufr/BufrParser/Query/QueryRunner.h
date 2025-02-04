/*
 * (C) Copyright 2022 NOAA/NWS/NCEP/EMC
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

#include "QuerySet.h"
#include "ResultSet.h"
#include "DataProvider/DataProvider.h"
#include "DataProvider/SubsetVariant.h"
#include "Target.h"

namespace Ingester {
namespace bufr {
    namespace __details
    {
        /// \brief BUFR messages are indexed according to start and stop values that are dependant
        /// on the message itself (the indexing is a property of the message). This object allows
        /// lets you make an array where the indexing is offset with respect to the actual position
        /// of the object in the array.
        template <typename T>
        class OffsetArray
        {
         public:
            OffsetArray(size_t startIdx, size_t endIdx)
                : offset_(startIdx)
            {
                data_.resize(endIdx - startIdx + 1);
            }

            T& operator[](size_t idx) { return data_[idx - offset_]; }

         private:
            std::vector<T> data_;
            size_t offset_;
        };

        /// \brief Masks used to make the processing of BUFR data more efficient. The aim is to skip
        /// branches without data we care about.
        struct ProcessingMasks {
            std::vector<bool> valueNodeMask;
            std::vector<bool> pathNodeMask;
        };
    }  // namespace __details

    /// \brief Manages the execution of queries against on a BUFR file.
    class QueryRunner
    {
     public:
        /// \brief Constructor.
        /// \param[in] querySet The set of queries to execute against the BUFR file.
        /// \param[in, out] resultSet The object used to store the accumulated collected data.
        /// \param[in] dataProvider The BUFR data provider to use.
        QueryRunner(const QuerySet& querySet,
                    ResultSet& resultSet,
                    const DataProviderType& dataProvider);
        void accumulate();

     private:
        const QuerySet querySet_;
        ResultSet& resultSet_;
        const DataProviderType& dataProvider_;

        std::unordered_map<SubsetVariant, Targets> targetCache_;
        std::unordered_map<SubsetVariant, std::shared_ptr<__details::ProcessingMasks>> maskCache_;
        std::unordered_map<SubsetVariant, std::unordered_map<std::string, std::string>> unitCache_;


        /// \brief Look for the list of targets for the currently active BUFR message subset that
        /// apply to the QuerySet and cache them. Processing mask information is also collected in
        /// order to make the data collection more efficient.
        /// \param[in, out] targets The list of targets to populate.
        /// \param[in, out] masks The processing masks to populate.
        void findTargets(Targets& targets,
                         std::shared_ptr<__details::ProcessingMasks>& masks);


        /// \brief Does the node idx correspond to an element you'd find in a query string (repeat
        /// or binary sequence)?
        /// \param[in] nodeIdx The node index to check.
        bool isQueryNode(int nodeIdx) const;


        /// \brief Accumulate the data for the currently open BUFR message subset.
        /// \param[in] targets The list of targets to collect for this subset.
        /// \param[in] masks The processing masks to use.
        /// \param[in, out] resultSet The object used to store the accumulated collected data.
        void collectData(Targets& targets,
                         std::shared_ptr<__details::ProcessingMasks> masks,
                         ResultSet& resultSet) const;


        /// \brief Given data counts and a filter specification this function creates the resulting
        ///        data vector.
        /// \param[in] srcData The source data vector.
        /// \param[in] origCounts The original data counts.
        /// \param[in] filter The filter specification.
        /// \return The resulting data vector after the filter is applied.
        std::vector<double> makeFilteredData(const std::vector<double>& srcData,
                                             const SeqCounts &origCounts,
                                             const std::vector<std::vector<size_t>> &filter)
                                                  const;

        /// \brief Recursive function that does the actual work of creating the filtered data
        ///        vector.
        /// \param[in] srcData The source data vector.
        /// \param[in] origCounts The original data counts.
        /// \param[in] filters The filter specification.
        /// \param[in, out] data The resulting data vector.
        /// \param[in, out] offset The current offset into the resulting data vector.
        /// \param[in] depth The current depth of the recursion.
        /// \param[in] skipResult If true, the result of the current recursion is not stored in the
        ///                       resulting data vector. This data is being filtered out.
        void _makeFilteredData(const std::vector<double>& srcData,
                               const SeqCounts& origCounts,
                               const std::vector<std::vector<size_t>>& filters,
                               std::vector<double>& data,
                               size_t& offset,
                               size_t depth,
                               bool skipResult = false) const;
    };
}  // namespace bufr
}  // namespace Ingester
