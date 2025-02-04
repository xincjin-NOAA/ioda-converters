/*
 * (C) Copyright 2022 NOAA/NWS/NCEP/EMC
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "DataProvider/DataProvider.h"
#include "Tokenizer.h"
#include "QueryParser.h"

namespace Ingester {
namespace bufr {
    /// \brief Compoenent information for the query path that leads to the target.
    struct TargetComponent
    {
        enum class Type
        {
            Subset,
            Binary,
            Repeat,
            Value,
            Unknown
        };

        std::shared_ptr<QueryComponent> queryComponent;
        size_t branch;
        Type type;


        /// \brief Check if this component adds a dimension to the data.
        bool addsDimension() const
        {
            return (type == Type::Subset || type == Type::Repeat) &&
                   (queryComponent->filter.empty() || queryComponent->filter.size() > 1);
        }

        /// \brief Sets the TargetComponent type based on the type of the BUFR query node type.
        /// \param bufrTyp The type of the BUFR query node.
        void setType(const Typ& bufrTyp)
        {
            static std::unordered_map<Typ, Type> typMap = {
                {Typ::Subset, Type::Subset},
                {Typ::DelayedRep, Type::Repeat},
                {Typ::FixedRep, Type::Repeat},
                {Typ::DelayedRepStacked, Type::Repeat},
                {Typ::DelayedBinary, Type::Binary},
                {Typ::Repeat, Type::Repeat},
                {Typ::StackedRepeat, Type::Repeat},
                {Typ::Sequence, Type::Repeat},
                {Typ::Number, Type::Value},
                {Typ::Character, Type::Value}
            };

            this->type = typMap.at(bufrTyp);
        }
    };

    typedef std::vector<TargetComponent> TargetComponents;

    /// \brief The information or Meta data for a BUFR field whose data we wish to capture when
    /// we execute a query.
    struct Target
    {
        std::string name;
        std::string queryStr;
        std::string unit;
        TypeInfo typeInfo;
        size_t nodeIdx;
        TargetComponents path;
        size_t numDimensions = 0;

        std::vector<Query> dimPaths;
        std::vector<int> exportDimIdxs;
        std::vector<int> seqPath;

        Target() = default;

        /// \brief Sets metadata for a target given the TargetComponents in the path to the target.
        ///        It not only sets the path but also sets the dimensioning paths, the sequence
        ///        paths and the idxs for the exported components (the ones that add dimensions).
        /// \param components The TargetComponents in the path to the target.
        void setPath(const TargetComponents& components)
        {
            exportDimIdxs = {};
            seqPath.reserve(components.size());
            exportDimIdxs.reserve(components.size());

            std::string currentPath;
            std::vector<std::shared_ptr<QueryComponent>> queryComponents;
            size_t componentIdx = 0;
            for (const auto& component : components)
            {
                queryComponents.push_back(component.queryComponent);

                if (component.addsDimension())
                {
                    numDimensions++;
                    if (component.type == TargetComponent::Type::Subset)
                    {
                        dimPaths.push_back(Query());
                    }
                    else
                    {
                        dimPaths.emplace_back(queryComponents);
                    }

                    exportDimIdxs.emplace_back(componentIdx);
                }

                if (component.type == TargetComponent::Type::Repeat ||
                    component.type == TargetComponent::Type::Binary)
                {
                    seqPath.push_back(component.branch);
                }

                componentIdx++;
            }

            path = components;
        }
    };

    typedef std::vector<std::shared_ptr<Target>> Targets;
}  // namespace bufr
}  // namespace Ingester
