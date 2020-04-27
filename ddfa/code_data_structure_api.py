# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = logging_call_api_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Union, TypeVar, Callable, Type, cast
from enum import Enum


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


@dataclass
class SerFilterArgument:
    value: Any
    values: Optional[List[Any]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerFilterArgument':
        assert isinstance(obj, dict)
        value = obj.get("value")
        values = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("values"))
        return SerFilterArgument(value, values)

    def to_dict(self) -> dict:
        result: dict = {}
        result["value"] = self.value
        result["values"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.values)
        return result


class SerFilterCompOp(Enum):
    EQUALS = "Equals"
    GREATER_EQUAL = "GreaterEqual"
    GREATER_THAN = "GreaterThan"
    IN = "In"
    LOWER_EQUAL = "LowerEqual"
    LOWER_THAN = "LowerThan"
    NOT_IN = "NotIn"


class SerFeatureValueType(Enum):
    BOOLEAN = "Boolean"
    CATEGORICAL = "Categorical"
    NUMERICAL = "Numerical"


@dataclass
class SerValueFilter:
    value_type: SerFeatureValueType
    arg: Optional[SerFilterArgument] = None
    comp_op: Optional[SerFilterCompOp] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerValueFilter':
        assert isinstance(obj, dict)
        value_type = SerFeatureValueType(obj.get("valueType"))
        arg = from_union([SerFilterArgument.from_dict, from_none], obj.get("arg"))
        comp_op = from_union([SerFilterCompOp, from_none], obj.get("compOp"))
        return SerValueFilter(value_type, arg, comp_op)

    def to_dict(self) -> dict:
        result: dict = {}
        result["valueType"] = to_enum(SerFeatureValueType, self.value_type)
        result["arg"] = from_union([lambda x: to_class(SerFilterArgument, x), from_none], self.arg)
        result["compOp"] = from_union([lambda x: to_enum(SerFilterCompOp, x), from_none], self.comp_op)
        return result


@dataclass
class SerLoggingCallCalculatedFeatureFilter:
    feature_name: str
    value_filter: SerValueFilter

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCallCalculatedFeatureFilter':
        assert isinstance(obj, dict)
        feature_name = from_str(obj.get("featureName"))
        value_filter = SerValueFilter.from_dict(obj.get("valueFilter"))
        return SerLoggingCallCalculatedFeatureFilter(feature_name, value_filter)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureName"] = from_str(self.feature_name)
        result["valueFilter"] = to_class(SerValueFilter, self.value_filter)
        return result


@dataclass
class SerLoggingCallExperimentResultMetricFilter:
    experiment_names: Optional[List[str]] = None
    metric_name: Optional[str] = None
    value_filter: Optional[SerValueFilter] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCallExperimentResultMetricFilter':
        assert isinstance(obj, dict)
        experiment_names = from_union([lambda x: from_list(from_str, x), from_none], obj.get("experimentNames"))
        metric_name = from_union([from_str, from_none], obj.get("metricName"))
        value_filter = from_union([SerValueFilter.from_dict, from_none], obj.get("valueFilter"))
        return SerLoggingCallExperimentResultMetricFilter(experiment_names, metric_name, value_filter)

    def to_dict(self) -> dict:
        result: dict = {}
        result["experimentNames"] = from_union([lambda x: from_list(from_str, x), from_none], self.experiment_names)
        result["metricName"] = from_union([from_str, from_none], self.metric_name)
        result["valueFilter"] = from_union([lambda x: to_class(SerValueFilter, x), from_none], self.value_filter)
        return result


class SerQualifier(Enum):
    EXISTS = "Exists"
    FOR_ALL = "ForAll"


@dataclass
class SerSymbolNameCalculatedFeatureFilter:
    feature_name: str
    value_filter: SerValueFilter

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolNameCalculatedFeatureFilter':
        assert isinstance(obj, dict)
        feature_name = from_str(obj.get("featureName"))
        value_filter = SerValueFilter.from_dict(obj.get("valueFilter"))
        return SerSymbolNameCalculatedFeatureFilter(feature_name, value_filter)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureName"] = from_str(self.feature_name)
        result["valueFilter"] = to_class(SerValueFilter, self.value_filter)
        return result


class SerExperimentResultConfusionType(Enum):
    FALSE_NEGATIVE = "FalseNegative"
    FALSE_POSITIVE = "FalsePositive"
    TRUE_NEGATIVE = "TrueNegative"
    TRUE_POSITIVE = "TruePositive"


@dataclass
class SerSymbolNameExperimentResultFilter:
    confusion_results: Optional[List[SerExperimentResultConfusionType]] = None
    experiment_names: Optional[List[str]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolNameExperimentResultFilter':
        assert isinstance(obj, dict)
        confusion_results = from_union([lambda x: from_list(SerExperimentResultConfusionType, x), from_none], obj.get("confusionResults"))
        experiment_names = from_union([lambda x: from_list(from_str, x), from_none], obj.get("experimentNames"))
        return SerSymbolNameExperimentResultFilter(confusion_results, experiment_names)

    def to_dict(self) -> dict:
        result: dict = {}
        result["confusionResults"] = from_union([lambda x: from_list(lambda x: to_enum(SerExperimentResultConfusionType, x), x), from_none], self.confusion_results)
        result["experimentNames"] = from_union([lambda x: from_list(from_str, x), from_none], self.experiment_names)
        return result


@dataclass
class SerSymbolNameFilter:
    calculated_feature: Optional[SerSymbolNameCalculatedFeatureFilter] = None
    experiment_result: Optional[SerSymbolNameExperimentResultFilter] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolNameFilter':
        assert isinstance(obj, dict)
        calculated_feature = from_union([SerSymbolNameCalculatedFeatureFilter.from_dict, from_none], obj.get("calculatedFeature"))
        experiment_result = from_union([SerSymbolNameExperimentResultFilter.from_dict, from_none], obj.get("experimentResult"))
        return SerSymbolNameFilter(calculated_feature, experiment_result)

    def to_dict(self) -> dict:
        result: dict = {}
        result["calculatedFeature"] = from_union([lambda x: to_class(SerSymbolNameCalculatedFeatureFilter, x), from_none], self.calculated_feature)
        result["experimentResult"] = from_union([lambda x: to_class(SerSymbolNameExperimentResultFilter, x), from_none], self.experiment_result)
        return result


@dataclass
class SerLoggingCallSymbolNameFilter:
    qualifier: SerQualifier
    symbol_name_filters: Optional[List[SerSymbolNameFilter]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCallSymbolNameFilter':
        assert isinstance(obj, dict)
        qualifier = SerQualifier(obj.get("qualifier"))
        symbol_name_filters = from_union([lambda x: from_list(SerSymbolNameFilter.from_dict, x), from_none], obj.get("symbolNameFilters"))
        return SerLoggingCallSymbolNameFilter(qualifier, symbol_name_filters)

    def to_dict(self) -> dict:
        result: dict = {}
        result["qualifier"] = to_enum(SerQualifier, self.qualifier)
        result["symbolNameFilters"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbolNameFilter, x), x), from_none], self.symbol_name_filters)
        return result


@dataclass
class SerLoggingCallFilter:
    calculated_feature: Optional[SerLoggingCallCalculatedFeatureFilter] = None
    experiment_result_metric: Optional[SerLoggingCallExperimentResultMetricFilter] = None
    symbol_name: Optional[SerLoggingCallSymbolNameFilter] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCallFilter':
        assert isinstance(obj, dict)
        calculated_feature = from_union([SerLoggingCallCalculatedFeatureFilter.from_dict, from_none], obj.get("calculatedFeature"))
        experiment_result_metric = from_union([SerLoggingCallExperimentResultMetricFilter.from_dict, from_none], obj.get("experimentResultMetric"))
        symbol_name = from_union([SerLoggingCallSymbolNameFilter.from_dict, from_none], obj.get("symbolName"))
        return SerLoggingCallFilter(calculated_feature, experiment_result_metric, symbol_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["calculatedFeature"] = from_union([lambda x: to_class(SerLoggingCallCalculatedFeatureFilter, x), from_none], self.calculated_feature)
        result["experimentResultMetric"] = from_union([lambda x: to_class(SerLoggingCallExperimentResultMetricFilter, x), from_none], self.experiment_result_metric)
        result["symbolName"] = from_union([lambda x: to_class(SerLoggingCallSymbolNameFilter, x), from_none], self.symbol_name)
        return result


@dataclass
class QDummyGetFiltes:
    arguments: Optional[Dict[str, Any]] = None
    q_dummy_get_filtes_return: Optional[List[SerLoggingCallFilter]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QDummyGetFiltes':
        assert isinstance(obj, dict)
        arguments = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("arguments"))
        q_dummy_get_filtes_return = from_union([lambda x: from_list(SerLoggingCallFilter.from_dict, x), from_none], obj.get("return"))
        return QDummyGetFiltes(arguments, q_dummy_get_filtes_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: from_list(lambda x: to_class(SerLoggingCallFilter, x), x), from_none], self.q_dummy_get_filtes_return)
        return result


@dataclass
class SerExperiment:
    name: str
    base_model: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None
    model_parameters: Optional[str] = None
    nr_epochs: Optional[int] = None
    number: Optional[int] = None
    sub_model: Optional[str] = None
    trained_on_dataset: Optional[str] = None
    train_output: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerExperiment':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        base_model = from_union([from_str, from_none], obj.get("baseModel"))
        description = from_union([from_str, from_none], obj.get("description"))
        link = from_union([from_str, from_none], obj.get("link"))
        model_parameters = from_union([from_str, from_none], obj.get("modelParameters"))
        nr_epochs = from_union([from_int, from_none], obj.get("nrEpochs"))
        number = from_union([from_int, from_none], obj.get("number"))
        sub_model = from_union([from_str, from_none], obj.get("subModel"))
        trained_on_dataset = from_union([from_str, from_none], obj.get("trainedOnDataset"))
        train_output = from_union([from_str, from_none], obj.get("trainOutput"))
        return SerExperiment(name, base_model, description, link, model_parameters, nr_epochs, number, sub_model, trained_on_dataset, train_output)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["baseModel"] = from_union([from_str, from_none], self.base_model)
        result["description"] = from_union([from_str, from_none], self.description)
        result["link"] = from_union([from_str, from_none], self.link)
        result["modelParameters"] = from_union([from_str, from_none], self.model_parameters)
        result["nrEpochs"] = from_union([from_int, from_none], self.nr_epochs)
        result["number"] = from_union([from_int, from_none], self.number)
        result["subModel"] = from_union([from_str, from_none], self.sub_model)
        result["trainedOnDataset"] = from_union([from_str, from_none], self.trained_on_dataset)
        result["trainOutput"] = from_union([from_str, from_none], self.train_output)
        return result


@dataclass
class QExperiments:
    arguments: Optional[Dict[str, Any]] = None
    q_experiments_return: Optional[List[SerExperiment]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QExperiments':
        assert isinstance(obj, dict)
        arguments = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("arguments"))
        q_experiments_return = from_union([lambda x: from_list(SerExperiment.from_dict, x), from_none], obj.get("return"))
        return QExperiments(arguments, q_experiments_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: from_list(lambda x: to_class(SerExperiment, x), x), from_none], self.q_experiments_return)
        return result


@dataclass
class QLoggingCallArguments:
    id: Any

    @staticmethod
    def from_dict(obj: Any) -> 'QLoggingCallArguments':
        assert isinstance(obj, dict)
        id = obj.get("id")
        return QLoggingCallArguments(id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = self.id
        return result


@dataclass
class SerLoggingCallCalculatedFeature:
    feature_name: str
    feature_value: Any

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCallCalculatedFeature':
        assert isinstance(obj, dict)
        feature_name = from_str(obj.get("featureName"))
        feature_value = obj.get("featureValue")
        return SerLoggingCallCalculatedFeature(feature_name, feature_value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureName"] = from_str(self.feature_name)
        result["featureValue"] = self.feature_value
        return result


@dataclass
class SerSymbolNameCalculatedFeatureValue:
    feature_name: str
    feature_value: Any

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolNameCalculatedFeatureValue':
        assert isinstance(obj, dict)
        feature_name = from_str(obj.get("featureName"))
        feature_value = obj.get("featureValue")
        return SerSymbolNameCalculatedFeatureValue(feature_name, feature_value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureName"] = from_str(self.feature_name)
        result["featureValue"] = self.feature_value
        return result


@dataclass
class SerSymbolNameCalculatedFeatures:
    symbol_name: str
    features_values: Optional[List[SerSymbolNameCalculatedFeatureValue]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolNameCalculatedFeatures':
        assert isinstance(obj, dict)
        symbol_name = from_str(obj.get("symbolName"))
        features_values = from_union([lambda x: from_list(SerSymbolNameCalculatedFeatureValue.from_dict, x), from_none], obj.get("featuresValues"))
        return SerSymbolNameCalculatedFeatures(symbol_name, features_values)

    def to_dict(self) -> dict:
        result: dict = {}
        result["symbolName"] = from_str(self.symbol_name)
        result["featuresValues"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbolNameCalculatedFeatureValue, x), x), from_none], self.features_values)
        return result


@dataclass
class SerCalculatedFeatures:
    logging_call_features: Optional[List[SerLoggingCallCalculatedFeature]] = None
    symbol_name_features: Optional[List[SerSymbolNameCalculatedFeatures]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerCalculatedFeatures':
        assert isinstance(obj, dict)
        logging_call_features = from_union([lambda x: from_list(SerLoggingCallCalculatedFeature.from_dict, x), from_none], obj.get("loggingCallFeatures"))
        symbol_name_features = from_union([lambda x: from_list(SerSymbolNameCalculatedFeatures.from_dict, x), from_none], obj.get("symbolNameFeatures"))
        return SerCalculatedFeatures(logging_call_features, symbol_name_features)

    def to_dict(self) -> dict:
        result: dict = {}
        result["loggingCallFeatures"] = from_union([lambda x: from_list(lambda x: to_class(SerLoggingCallCalculatedFeature, x), x), from_none], self.logging_call_features)
        result["symbolNameFeatures"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbolNameCalculatedFeatures, x), x), from_none], self.symbol_name_features)
        return result


@dataclass
class SerPosition:
    col: int
    line: int

    @staticmethod
    def from_dict(obj: Any) -> 'SerPosition':
        assert isinstance(obj, dict)
        col = from_int(obj.get("col"))
        line = from_int(obj.get("line"))
        return SerPosition(col, line)

    def to_dict(self) -> dict:
        result: dict = {}
        result["col"] = from_int(self.col)
        result["line"] = from_int(self.line)
        return result


@dataclass
class SerPositionRange:
    begin: SerPosition
    end: SerPosition

    @staticmethod
    def from_dict(obj: Any) -> 'SerPositionRange':
        assert isinstance(obj, dict)
        begin = SerPosition.from_dict(obj.get("begin"))
        end = SerPosition.from_dict(obj.get("end"))
        return SerPositionRange(begin, end)

    def to_dict(self) -> dict:
        result: dict = {}
        result["begin"] = to_class(SerPosition, self.begin)
        result["end"] = to_class(SerPosition, self.end)
        return result


class SerTokenKind(Enum):
    COMMENT = "Comment"
    EOL = "EOL"
    IDENTIFIER = "Identifier"
    KEYWORD = "Keyword"
    LITERAL = "Literal"
    OPERATOR = "Operator"
    SEPARATOR = "Separator"
    WHITESPACE_NO_EOL = "WhitespaceNoEOL"


class SerTokenOperatorKind(Enum):
    ASSIGN = "Assign"
    ASSIGN_BINARY_AND = "AssignBinaryAnd"
    ASSIGN_BINARY_OR = "AssignBinaryOr"
    ASSIGN_DIVIDE = "AssignDivide"
    ASSIGN_LEFT_SHIFT = "AssignLeftShift"
    ASSIGN_MINUS = "AssignMinus"
    ASSIGN_MULTIPLY = "AssignMultiply"
    ASSIGN_PLUS = "AssignPlus"
    ASSIGN_REMINDER = "AssignReminder"
    ASSIGN_SIGNED_RIGHT_SHIFT = "AssignSignedRightShift"
    ASSIGN_UNSIGNED_RIGHT_SHIFT = "AssignUnsignedRightShift"
    ASSIGN_XOR = "AssignXOR"
    BINARY_AND = "BinaryAnd"
    BINARY_OR = "BinaryOr"
    BITWISE_COMPLEMENT = "BitwiseComplement"
    CONDITION_EXPR = "ConditionExpr"
    CONDITION_EXPR_ELSE = "ConditionExprElse"
    DECREMENT = "Decrement"
    DIVIDE = "Divide"
    EQUALS = "Equals"
    EXPAND = "Expand"
    GREATER = "Greater"
    GREATER_EQUALS = "GreaterEquals"
    INCREMENT = "Increment"
    LEFT_SHIFT = "LeftShift"
    LESS = "Less"
    LESS_EQUALS = "LessEquals"
    LOGICAL_AND = "LogicalAnd"
    LOGICAL_COMPLEMENT = "LogicalComplement"
    LOGICAL_OR = "LogicalOr"
    MINUS = "Minus"
    MULTIPLY = "Multiply"
    NOT_EQUALS = "NotEquals"
    PLUS = "Plus"
    REMINDER = "Reminder"
    SIGNED_RIGHT_SHIFT = "SignedRightShift"
    UNSIGNED_RIGHT_SHIFT = "UnsignedRightShift"
    XOR = "XOR"


class SerTokenSeparatorKind(Enum):
    AT = "AT"
    COMMA = "COMMA"
    DOTAIM = "DOTAIM"
    EOI = "EOI"
    LCPAR = "LCPAR"
    LPAR = "LPAR"
    LRPAR = "LRPAR"
    PERIOD = "PERIOD"
    RCPAR = "RCPAR"
    RPAR = "RPAR"
    RRPAR = "RRPAR"


@dataclass
class SerToken:
    kind: SerTokenKind
    identifier_idx: Optional[int] = None
    operator: Optional[SerTokenOperatorKind] = None
    separator: Optional[SerTokenSeparatorKind] = None
    symbol_idx: Optional[int] = None
    text: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerToken':
        assert isinstance(obj, dict)
        kind = SerTokenKind(obj.get("kind"))
        identifier_idx = from_union([from_int, from_none], obj.get("identifierIdx"))
        operator = from_union([SerTokenOperatorKind, from_none], obj.get("operator"))
        separator = from_union([SerTokenSeparatorKind, from_none], obj.get("separator"))
        symbol_idx = from_union([from_int, from_none], obj.get("symbolIdx"))
        text = from_union([from_str, from_none], obj.get("text"))
        return SerToken(kind, identifier_idx, operator, separator, symbol_idx, text)

    def to_dict(self) -> dict:
        result: dict = {}
        result["kind"] = to_enum(SerTokenKind, self.kind)
        result["identifierIdx"] = from_union([from_int, from_none], self.identifier_idx)
        result["operator"] = from_union([lambda x: to_enum(SerTokenOperatorKind, x), from_none], self.operator)
        result["separator"] = from_union([lambda x: to_enum(SerTokenSeparatorKind, x), from_none], self.separator)
        result["symbolIdx"] = from_union([from_int, from_none], self.symbol_idx)
        result["text"] = from_union([from_str, from_none], self.text)
        return result


@dataclass
class SerCodeSnippet:
    code: str
    position: SerPositionRange
    tokenized: Optional[List[SerToken]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerCodeSnippet':
        assert isinstance(obj, dict)
        code = from_str(obj.get("code"))
        position = SerPositionRange.from_dict(obj.get("position"))
        tokenized = from_union([lambda x: from_list(SerToken.from_dict, x), from_none], obj.get("tokenized"))
        return SerCodeSnippet(code, position, tokenized)

    def to_dict(self) -> dict:
        result: dict = {}
        result["code"] = from_str(self.code)
        result["position"] = to_class(SerPositionRange, self.position)
        result["tokenized"] = from_union([lambda x: from_list(lambda x: to_class(SerToken, x), x), from_none], self.tokenized)
        return result


@dataclass
class SerNameSymbolAssignment:
    name: str
    position_range: SerPositionRange
    scope_depth: int
    ast_node_idx: Optional[int] = None
    code: Optional[str] = None
    scope_hash: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerNameSymbolAssignment':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        position_range = SerPositionRange.from_dict(obj.get("positionRange"))
        scope_depth = from_int(obj.get("scopeDepth"))
        ast_node_idx = from_union([from_int, from_none], obj.get("astNodeIdx"))
        code = from_union([from_str, from_none], obj.get("code"))
        scope_hash = from_union([from_str, from_none], obj.get("scopeHash"))
        return SerNameSymbolAssignment(name, position_range, scope_depth, ast_node_idx, code, scope_hash)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["positionRange"] = to_class(SerPositionRange, self.position_range)
        result["scopeDepth"] = from_int(self.scope_depth)
        result["astNodeIdx"] = from_union([from_int, from_none], self.ast_node_idx)
        result["code"] = from_union([from_str, from_none], self.code)
        result["scopeHash"] = from_union([from_str, from_none], self.scope_hash)
        return result


class SerBranchEscapeInstType(Enum):
    BREAK = "Break"
    CONTINUE = "Continue"
    EXIT = "Exit"
    RETURN = "Return"


class SerNameSymbolDefinitionType(Enum):
    CLASS_MEMBER = "ClassMember"
    CLASS_METHOD = "ClassMethod"
    LOCAL_PARAM = "LocalParam"
    LOCAL_VARIABLE = "LocalVariable"


@dataclass
class SerNameSymbolDefinition:
    def_type: SerNameSymbolDefinitionType
    has_initializer: bool
    name: str
    position_range: SerPositionRange
    scope_depth: int
    ast_node_idx: Optional[int] = None
    code: Optional[str] = None
    scope_hash: Optional[str] = None
    type_name: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerNameSymbolDefinition':
        assert isinstance(obj, dict)
        def_type = SerNameSymbolDefinitionType(obj.get("defType"))
        has_initializer = from_bool(obj.get("hasInitializer"))
        name = from_str(obj.get("name"))
        position_range = SerPositionRange.from_dict(obj.get("positionRange"))
        scope_depth = from_int(obj.get("scopeDepth"))
        ast_node_idx = from_union([from_int, from_none], obj.get("astNodeIdx"))
        code = from_union([from_str, from_none], obj.get("code"))
        scope_hash = from_union([from_str, from_none], obj.get("scopeHash"))
        type_name = from_union([from_str, from_none], obj.get("typeName"))
        return SerNameSymbolDefinition(def_type, has_initializer, name, position_range, scope_depth, ast_node_idx, code, scope_hash, type_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["defType"] = to_enum(SerNameSymbolDefinitionType, self.def_type)
        result["hasInitializer"] = from_bool(self.has_initializer)
        result["name"] = from_str(self.name)
        result["positionRange"] = to_class(SerPositionRange, self.position_range)
        result["scopeDepth"] = from_int(self.scope_depth)
        result["astNodeIdx"] = from_union([from_int, from_none], self.ast_node_idx)
        result["code"] = from_union([from_str, from_none], self.code)
        result["scopeHash"] = from_union([from_str, from_none], self.scope_hash)
        result["typeName"] = from_union([from_str, from_none], self.type_name)
        return result


class SerScopeType(Enum):
    BLOCK_STMT = "BlockStmt"
    CATCH_CLAUSE = "CatchClause"
    CLASS = "Class"
    FOR_EACH_STMT = "ForEachStmt"
    FOR_STMT = "ForStmt"
    IF_STMT = "IfStmt"
    METHOD = "Method"
    TRY_STMT = "TryStmt"
    WHILE_STMT = "WhileStmt"


@dataclass
class SerContextualScope:
    depth: int
    position_range: SerPositionRange
    type: SerScopeType
    assignments: Optional[List[SerNameSymbolAssignment]] = None
    ast_node_idx: Optional[int] = None
    branch_escape_inst_types: Optional[List[SerBranchEscapeInstType]] = None
    class_members_def: Optional[List[SerNameSymbolDefinition]] = None
    class_methods_def: Optional[List[SerNameSymbolDefinition]] = None
    hash: Optional[str] = None
    local_params_def: Optional[List[SerNameSymbolDefinition]] = None
    local_variables_def: Optional[List[SerNameSymbolDefinition]] = None
    name: Optional[str] = None
    nr_log_calls: Optional[int] = None
    symbol_names_occurred_after: Optional[List[str]] = None
    symbol_names_occurred_before: Optional[List[str]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerContextualScope':
        assert isinstance(obj, dict)
        depth = from_int(obj.get("depth"))
        position_range = SerPositionRange.from_dict(obj.get("positionRange"))
        type = SerScopeType(obj.get("type"))
        assignments = from_union([lambda x: from_list(SerNameSymbolAssignment.from_dict, x), from_none], obj.get("assignments"))
        ast_node_idx = from_union([from_int, from_none], obj.get("astNodeIdx"))
        branch_escape_inst_types = from_union([lambda x: from_list(SerBranchEscapeInstType, x), from_none], obj.get("branchEscapeInstTypes"))
        class_members_def = from_union([lambda x: from_list(SerNameSymbolDefinition.from_dict, x), from_none], obj.get("classMembersDef"))
        class_methods_def = from_union([lambda x: from_list(SerNameSymbolDefinition.from_dict, x), from_none], obj.get("classMethodsDef"))
        hash = from_union([from_str, from_none], obj.get("hash"))
        local_params_def = from_union([lambda x: from_list(SerNameSymbolDefinition.from_dict, x), from_none], obj.get("localParamsDef"))
        local_variables_def = from_union([lambda x: from_list(SerNameSymbolDefinition.from_dict, x), from_none], obj.get("localVariablesDef"))
        name = from_union([from_str, from_none], obj.get("name"))
        nr_log_calls = from_union([from_int, from_none], obj.get("nrLogCalls"))
        symbol_names_occurred_after = from_union([lambda x: from_list(from_str, x), from_none], obj.get("symbolNamesOccurredAfter"))
        symbol_names_occurred_before = from_union([lambda x: from_list(from_str, x), from_none], obj.get("symbolNamesOccurredBefore"))
        return SerContextualScope(depth, position_range, type, assignments, ast_node_idx, branch_escape_inst_types, class_members_def, class_methods_def, hash, local_params_def, local_variables_def, name, nr_log_calls, symbol_names_occurred_after, symbol_names_occurred_before)

    def to_dict(self) -> dict:
        result: dict = {}
        result["depth"] = from_int(self.depth)
        result["positionRange"] = to_class(SerPositionRange, self.position_range)
        result["type"] = to_enum(SerScopeType, self.type)
        result["assignments"] = from_union([lambda x: from_list(lambda x: to_class(SerNameSymbolAssignment, x), x), from_none], self.assignments)
        result["astNodeIdx"] = from_union([from_int, from_none], self.ast_node_idx)
        result["branchEscapeInstTypes"] = from_union([lambda x: from_list(lambda x: to_enum(SerBranchEscapeInstType, x), x), from_none], self.branch_escape_inst_types)
        result["classMembersDef"] = from_union([lambda x: from_list(lambda x: to_class(SerNameSymbolDefinition, x), x), from_none], self.class_members_def)
        result["classMethodsDef"] = from_union([lambda x: from_list(lambda x: to_class(SerNameSymbolDefinition, x), x), from_none], self.class_methods_def)
        result["hash"] = from_union([from_str, from_none], self.hash)
        result["localParamsDef"] = from_union([lambda x: from_list(lambda x: to_class(SerNameSymbolDefinition, x), x), from_none], self.local_params_def)
        result["localVariablesDef"] = from_union([lambda x: from_list(lambda x: to_class(SerNameSymbolDefinition, x), x), from_none], self.local_variables_def)
        result["name"] = from_union([from_str, from_none], self.name)
        result["nrLogCalls"] = from_union([from_int, from_none], self.nr_log_calls)
        result["symbolNamesOccurredAfter"] = from_union([lambda x: from_list(from_str, x), from_none], self.symbol_names_occurred_after)
        result["symbolNamesOccurredBefore"] = from_union([lambda x: from_list(from_str, x), from_none], self.symbol_names_occurred_before)
        return result


@dataclass
class SerContextualScopesContainer:
    scopes: Optional[List[SerContextualScope]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerContextualScopesContainer':
        assert isinstance(obj, dict)
        scopes = from_union([lambda x: from_list(SerContextualScope.from_dict, x), from_none], obj.get("scopes"))
        return SerContextualScopesContainer(scopes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["scopes"] = from_union([lambda x: from_list(lambda x: to_class(SerContextualScope, x), x), from_none], self.scopes)
        return result


@dataclass
class SerTokenMetric:
    false_negatives: int
    false_positives: int
    nr_ground_true_tokens: int
    nr_predicted_tokens: int
    true_negatives: int
    true_positives: int

    @staticmethod
    def from_dict(obj: Any) -> 'SerTokenMetric':
        assert isinstance(obj, dict)
        false_negatives = from_int(obj.get("falseNegatives"))
        false_positives = from_int(obj.get("falsePositives"))
        nr_ground_true_tokens = from_int(obj.get("nrGroundTrueTokens"))
        nr_predicted_tokens = from_int(obj.get("nrPredictedTokens"))
        true_negatives = from_int(obj.get("trueNegatives"))
        true_positives = from_int(obj.get("truePositives"))
        return SerTokenMetric(false_negatives, false_positives, nr_ground_true_tokens, nr_predicted_tokens, true_negatives, true_positives)

    def to_dict(self) -> dict:
        result: dict = {}
        result["falseNegatives"] = from_int(self.false_negatives)
        result["falsePositives"] = from_int(self.false_positives)
        result["nrGroundTrueTokens"] = from_int(self.nr_ground_true_tokens)
        result["nrPredictedTokens"] = from_int(self.nr_predicted_tokens)
        result["trueNegatives"] = from_int(self.true_negatives)
        result["truePositives"] = from_int(self.true_positives)
        return result


@dataclass
class SerExperimentResults:
    experiment_name: str
    token_metric: SerTokenMetric
    logging_call_hash: Optional[str] = None
    top_k_predicted_names: Optional[List[List[str]]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerExperimentResults':
        assert isinstance(obj, dict)
        experiment_name = from_str(obj.get("experimentName"))
        token_metric = SerTokenMetric.from_dict(obj.get("tokenMetric"))
        logging_call_hash = from_union([from_str, from_none], obj.get("loggingCallHash"))
        top_k_predicted_names = from_union([lambda x: from_list(lambda x: from_list(from_str, x), x), from_none], obj.get("topKPredictedNames"))
        return SerExperimentResults(experiment_name, token_metric, logging_call_hash, top_k_predicted_names)

    def to_dict(self) -> dict:
        result: dict = {}
        result["experimentName"] = from_str(self.experiment_name)
        result["tokenMetric"] = to_class(SerTokenMetric, self.token_metric)
        result["loggingCallHash"] = from_union([from_str, from_none], self.logging_call_hash)
        result["topKPredictedNames"] = from_union([lambda x: from_list(lambda x: from_list(from_str, x), x), from_none], self.top_k_predicted_names)
        return result


class SerDataFoldType(Enum):
    TEST = "Test"
    TRAIN = "Train"
    VALIDATION = "Validation"


@dataclass
class SerDatasetName:
    data_fold: SerDataFoldType
    name: str
    project_name: str

    @staticmethod
    def from_dict(obj: Any) -> 'SerDatasetName':
        assert isinstance(obj, dict)
        data_fold = SerDataFoldType(obj.get("dataFold"))
        name = from_str(obj.get("name"))
        project_name = from_str(obj.get("projectName"))
        return SerDatasetName(data_fold, name, project_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["dataFold"] = to_enum(SerDataFoldType, self.data_fold)
        result["name"] = from_str(self.name)
        result["projectName"] = from_str(self.project_name)
        return result


@dataclass
class SerMethodRef:
    code_filepath: str
    dataset_name: SerDatasetName
    hash: str
    name: str
    position: SerPositionRange
    class_name: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerMethodRef':
        assert isinstance(obj, dict)
        code_filepath = from_str(obj.get("codeFilepath"))
        dataset_name = SerDatasetName.from_dict(obj.get("datasetName"))
        hash = from_str(obj.get("hash"))
        name = from_str(obj.get("name"))
        position = SerPositionRange.from_dict(obj.get("position"))
        class_name = from_union([from_str, from_none], obj.get("className"))
        return SerMethodRef(code_filepath, dataset_name, hash, name, position, class_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["codeFilepath"] = from_str(self.code_filepath)
        result["datasetName"] = to_class(SerDatasetName, self.dataset_name)
        result["hash"] = from_str(self.hash)
        result["name"] = from_str(self.name)
        result["position"] = to_class(SerPositionRange, self.position)
        result["className"] = from_union([from_str, from_none], self.class_name)
        return result


@dataclass
class SerNameSymbolOccurrence:
    name: str
    definition: Optional[SerNameSymbolDefinition] = None
    last_assignment: Optional[SerNameSymbolAssignment] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerNameSymbolOccurrence':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        definition = from_union([SerNameSymbolDefinition.from_dict, from_none], obj.get("definition"))
        last_assignment = from_union([SerNameSymbolAssignment.from_dict, from_none], obj.get("lastAssignment"))
        return SerNameSymbolOccurrence(name, definition, last_assignment)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["definition"] = from_union([lambda x: to_class(SerNameSymbolDefinition, x), from_none], self.definition)
        result["lastAssignment"] = from_union([lambda x: to_class(SerNameSymbolAssignment, x), from_none], self.last_assignment)
        return result


@dataclass
class SerLoggingCall:
    code: SerCodeSnippet
    contextual_scopes: SerContextualScopesContainer
    hash: str
    method_ref: SerMethodRef
    ast_node_idx: Optional[int] = None
    calculated_features: Optional[SerCalculatedFeatures] = None
    experiments_results: Optional[List[SerExperimentResults]] = None
    inner_method_left_context_tokenized: Optional[List[str]] = None
    inner_method_right_context_tokenized: Optional[List[str]] = None
    lines_dist_from_closest_log_call_in_inner_scope: Optional[int] = None
    names_used_in_log_and_found_in_inner_method_scope: Optional[List[str]] = None
    name_symbols_used_in_log: Optional[List[SerNameSymbolOccurrence]] = None
    pdg_node_idx: Optional[int] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCall':
        assert isinstance(obj, dict)
        code = SerCodeSnippet.from_dict(obj.get("code"))
        contextual_scopes = SerContextualScopesContainer.from_dict(obj.get("contextualScopes"))
        hash = from_str(obj.get("hash"))
        method_ref = SerMethodRef.from_dict(obj.get("methodRef"))
        ast_node_idx = from_union([from_int, from_none], obj.get("astNodeIdx"))
        calculated_features = from_union([SerCalculatedFeatures.from_dict, from_none], obj.get("calculatedFeatures"))
        experiments_results = from_union([lambda x: from_list(SerExperimentResults.from_dict, x), from_none], obj.get("experimentsResults"))
        inner_method_left_context_tokenized = from_union([lambda x: from_list(from_str, x), from_none], obj.get("innerMethodLeftContextTokenized"))
        inner_method_right_context_tokenized = from_union([lambda x: from_list(from_str, x), from_none], obj.get("innerMethodRightContextTokenized"))
        lines_dist_from_closest_log_call_in_inner_scope = from_union([from_int, from_none], obj.get("linesDistFromClosestLogCallInInnerScope"))
        names_used_in_log_and_found_in_inner_method_scope = from_union([lambda x: from_list(from_str, x), from_none], obj.get("namesUsedInLogAndFoundInInnerMethodScope"))
        name_symbols_used_in_log = from_union([lambda x: from_list(SerNameSymbolOccurrence.from_dict, x), from_none], obj.get("nameSymbolsUsedInLog"))
        pdg_node_idx = from_union([from_int, from_none], obj.get("pdgNodeIdx"))
        return SerLoggingCall(code, contextual_scopes, hash, method_ref, ast_node_idx, calculated_features, experiments_results, inner_method_left_context_tokenized, inner_method_right_context_tokenized, lines_dist_from_closest_log_call_in_inner_scope, names_used_in_log_and_found_in_inner_method_scope, name_symbols_used_in_log, pdg_node_idx)

    def to_dict(self) -> dict:
        result: dict = {}
        result["code"] = to_class(SerCodeSnippet, self.code)
        result["contextualScopes"] = to_class(SerContextualScopesContainer, self.contextual_scopes)
        result["hash"] = from_str(self.hash)
        result["methodRef"] = to_class(SerMethodRef, self.method_ref)
        result["astNodeIdx"] = from_union([from_int, from_none], self.ast_node_idx)
        result["calculatedFeatures"] = from_union([lambda x: to_class(SerCalculatedFeatures, x), from_none], self.calculated_features)
        result["experimentsResults"] = from_union([lambda x: from_list(lambda x: to_class(SerExperimentResults, x), x), from_none], self.experiments_results)
        result["innerMethodLeftContextTokenized"] = from_union([lambda x: from_list(from_str, x), from_none], self.inner_method_left_context_tokenized)
        result["innerMethodRightContextTokenized"] = from_union([lambda x: from_list(from_str, x), from_none], self.inner_method_right_context_tokenized)
        result["linesDistFromClosestLogCallInInnerScope"] = from_union([from_int, from_none], self.lines_dist_from_closest_log_call_in_inner_scope)
        result["namesUsedInLogAndFoundInInnerMethodScope"] = from_union([lambda x: from_list(from_str, x), from_none], self.names_used_in_log_and_found_in_inner_method_scope)
        result["nameSymbolsUsedInLog"] = from_union([lambda x: from_list(lambda x: to_class(SerNameSymbolOccurrence, x), x), from_none], self.name_symbols_used_in_log)
        result["pdgNodeIdx"] = from_union([from_int, from_none], self.pdg_node_idx)
        return result


@dataclass
class QLoggingCall:
    arguments: Optional[QLoggingCallArguments] = None
    q_logging_call_return: Optional[SerLoggingCall] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QLoggingCall':
        assert isinstance(obj, dict)
        arguments = from_union([QLoggingCallArguments.from_dict, from_none], obj.get("arguments"))
        q_logging_call_return = from_union([SerLoggingCall.from_dict, from_none], obj.get("return"))
        return QLoggingCall(arguments, q_logging_call_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: to_class(QLoggingCallArguments, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: to_class(SerLoggingCall, x), from_none], self.q_logging_call_return)
        return result


class SerFeatureOfType(Enum):
    LOGGING_CALL = "LoggingCall"
    SYMBOL_NAME = "SymbolName"


@dataclass
class SerLoggingCallFeature:
    feature_name: str
    feature_of_type: SerFeatureOfType
    feature_value_type: SerFeatureValueType
    max_value: Any
    min_value: Any
    options: Optional[List[Any]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCallFeature':
        assert isinstance(obj, dict)
        feature_name = from_str(obj.get("featureName"))
        feature_of_type = SerFeatureOfType(obj.get("featureOfType"))
        feature_value_type = SerFeatureValueType(obj.get("featureValueType"))
        max_value = obj.get("maxValue")
        min_value = obj.get("minValue")
        options = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("options"))
        return SerLoggingCallFeature(feature_name, feature_of_type, feature_value_type, max_value, min_value, options)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureName"] = from_str(self.feature_name)
        result["featureOfType"] = to_enum(SerFeatureOfType, self.feature_of_type)
        result["featureValueType"] = to_enum(SerFeatureValueType, self.feature_value_type)
        result["maxValue"] = self.max_value
        result["minValue"] = self.min_value
        result["options"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.options)
        return result


@dataclass
class QLoggingCallFeatures:
    arguments: Optional[Dict[str, Any]] = None
    q_logging_call_features_return: Optional[List[SerLoggingCallFeature]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QLoggingCallFeatures':
        assert isinstance(obj, dict)
        arguments = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("arguments"))
        q_logging_call_features_return = from_union([lambda x: from_list(SerLoggingCallFeature.from_dict, x), from_none], obj.get("return"))
        return QLoggingCallFeatures(arguments, q_logging_call_features_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: from_list(lambda x: to_class(SerLoggingCallFeature, x), x), from_none], self.q_logging_call_features_return)
        return result


@dataclass
class SerBooleanValueStatistics:
    nr_falses: Optional[int] = None
    nr_trues: Optional[int] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerBooleanValueStatistics':
        assert isinstance(obj, dict)
        nr_falses = from_union([from_int, from_none], obj.get("nrFalses"))
        nr_trues = from_union([from_int, from_none], obj.get("nrTrues"))
        return SerBooleanValueStatistics(nr_falses, nr_trues)

    def to_dict(self) -> dict:
        result: dict = {}
        result["nrFalses"] = from_union([from_int, from_none], self.nr_falses)
        result["nrTrues"] = from_union([from_int, from_none], self.nr_trues)
        return result


@dataclass
class SerCategoryOccurrences:
    category: str
    occurrences: int

    @staticmethod
    def from_dict(obj: Any) -> 'SerCategoryOccurrences':
        assert isinstance(obj, dict)
        category = from_str(obj.get("category"))
        occurrences = from_int(obj.get("occurrences"))
        return SerCategoryOccurrences(category, occurrences)

    def to_dict(self) -> dict:
        result: dict = {}
        result["category"] = from_str(self.category)
        result["occurrences"] = from_int(self.occurrences)
        return result


@dataclass
class SerCategoricalValueStatistics:
    nr_occurrences_per_category: Optional[List[SerCategoryOccurrences]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerCategoricalValueStatistics':
        assert isinstance(obj, dict)
        nr_occurrences_per_category = from_union([lambda x: from_list(SerCategoryOccurrences.from_dict, x), from_none], obj.get("nrOccurrencesPerCategory"))
        return SerCategoricalValueStatistics(nr_occurrences_per_category)

    def to_dict(self) -> dict:
        result: dict = {}
        result["nrOccurrencesPerCategory"] = from_union([lambda x: from_list(lambda x: to_class(SerCategoryOccurrences, x), x), from_none], self.nr_occurrences_per_category)
        return result


@dataclass
class SerStatisticPercentile:
    percent: int
    percentile: int

    @staticmethod
    def from_dict(obj: Any) -> 'SerStatisticPercentile':
        assert isinstance(obj, dict)
        percent = from_int(obj.get("percent"))
        percentile = from_int(obj.get("percentile"))
        return SerStatisticPercentile(percent, percentile)

    def to_dict(self) -> dict:
        result: dict = {}
        result["percent"] = from_int(self.percent)
        result["percentile"] = from_int(self.percentile)
        return result


@dataclass
class SerNumericalValueStatistics:
    avg: int
    max: int
    median: int
    min: int
    stddev: int
    percentiles: Optional[List[SerStatisticPercentile]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerNumericalValueStatistics':
        assert isinstance(obj, dict)
        avg = from_int(obj.get("avg"))
        max = from_int(obj.get("max"))
        median = from_int(obj.get("median"))
        min = from_int(obj.get("min"))
        stddev = from_int(obj.get("stddev"))
        percentiles = from_union([lambda x: from_list(SerStatisticPercentile.from_dict, x), from_none], obj.get("percentiles"))
        return SerNumericalValueStatistics(avg, max, median, min, stddev, percentiles)

    def to_dict(self) -> dict:
        result: dict = {}
        result["avg"] = from_int(self.avg)
        result["max"] = from_int(self.max)
        result["median"] = from_int(self.median)
        result["min"] = from_int(self.min)
        result["stddev"] = from_int(self.stddev)
        result["percentiles"] = from_union([lambda x: from_list(lambda x: to_class(SerStatisticPercentile, x), x), from_none], self.percentiles)
        return result


@dataclass
class SerCalculatedFeatureStatistics:
    feature_name: str
    boolean_value_statistics: Optional[SerBooleanValueStatistics] = None
    categorical_value_statistics: Optional[SerCategoricalValueStatistics] = None
    numerical_value_statistics: Optional[SerNumericalValueStatistics] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerCalculatedFeatureStatistics':
        assert isinstance(obj, dict)
        feature_name = from_str(obj.get("featureName"))
        boolean_value_statistics = from_union([SerBooleanValueStatistics.from_dict, from_none], obj.get("booleanValueStatistics"))
        categorical_value_statistics = from_union([SerCategoricalValueStatistics.from_dict, from_none], obj.get("categoricalValueStatistics"))
        numerical_value_statistics = from_union([SerNumericalValueStatistics.from_dict, from_none], obj.get("numericalValueStatistics"))
        return SerCalculatedFeatureStatistics(feature_name, boolean_value_statistics, categorical_value_statistics, numerical_value_statistics)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureName"] = from_str(self.feature_name)
        result["booleanValueStatistics"] = from_union([lambda x: to_class(SerBooleanValueStatistics, x), from_none], self.boolean_value_statistics)
        result["categoricalValueStatistics"] = from_union([lambda x: to_class(SerCategoricalValueStatistics, x), from_none], self.categorical_value_statistics)
        result["numericalValueStatistics"] = from_union([lambda x: to_class(SerNumericalValueStatistics, x), from_none], self.numerical_value_statistics)
        return result


@dataclass
class SerTokenMetricStatistics:
    f1: int
    fn: int
    fp: int
    nr_logging_calls: int
    precision: int
    recall: int
    total_nr_ground_true_tokens: int
    total_nr_predicted_tokens: int
    tp: int

    @staticmethod
    def from_dict(obj: Any) -> 'SerTokenMetricStatistics':
        assert isinstance(obj, dict)
        f1 = from_int(obj.get("f1"))
        fn = from_int(obj.get("fn"))
        fp = from_int(obj.get("fp"))
        nr_logging_calls = from_int(obj.get("nrLoggingCalls"))
        precision = from_int(obj.get("precision"))
        recall = from_int(obj.get("recall"))
        total_nr_ground_true_tokens = from_int(obj.get("totalNrGroundTrueTokens"))
        total_nr_predicted_tokens = from_int(obj.get("totalNrPredictedTokens"))
        tp = from_int(obj.get("tp"))
        return SerTokenMetricStatistics(f1, fn, fp, nr_logging_calls, precision, recall, total_nr_ground_true_tokens, total_nr_predicted_tokens, tp)

    def to_dict(self) -> dict:
        result: dict = {}
        result["f1"] = from_int(self.f1)
        result["fn"] = from_int(self.fn)
        result["fp"] = from_int(self.fp)
        result["nrLoggingCalls"] = from_int(self.nr_logging_calls)
        result["precision"] = from_int(self.precision)
        result["recall"] = from_int(self.recall)
        result["totalNrGroundTrueTokens"] = from_int(self.total_nr_ground_true_tokens)
        result["totalNrPredictedTokens"] = from_int(self.total_nr_predicted_tokens)
        result["tp"] = from_int(self.tp)
        return result


@dataclass
class SerExperimentResultsStatistics:
    experiment_name: str
    token_metric_statistics: SerTokenMetricStatistics

    @staticmethod
    def from_dict(obj: Any) -> 'SerExperimentResultsStatistics':
        assert isinstance(obj, dict)
        experiment_name = from_str(obj.get("experimentName"))
        token_metric_statistics = SerTokenMetricStatistics.from_dict(obj.get("tokenMetricStatistics"))
        return SerExperimentResultsStatistics(experiment_name, token_metric_statistics)

    def to_dict(self) -> dict:
        result: dict = {}
        result["experimentName"] = from_str(self.experiment_name)
        result["tokenMetricStatistics"] = to_class(SerTokenMetricStatistics, self.token_metric_statistics)
        return result


@dataclass
class SerLoggingCallsStatistics:
    nr_filtered_records: int
    total_nr_records_in_dataset: int
    calculated_features_statistics: Optional[List[SerCalculatedFeatureStatistics]] = None
    experiments_results_statistics: Optional[List[SerExperimentResultsStatistics]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerLoggingCallsStatistics':
        assert isinstance(obj, dict)
        nr_filtered_records = from_int(obj.get("nrFilteredRecords"))
        total_nr_records_in_dataset = from_int(obj.get("totalNrRecordsInDataset"))
        calculated_features_statistics = from_union([lambda x: from_list(SerCalculatedFeatureStatistics.from_dict, x), from_none], obj.get("calculatedFeaturesStatistics"))
        experiments_results_statistics = from_union([lambda x: from_list(SerExperimentResultsStatistics.from_dict, x), from_none], obj.get("experimentsResultsStatistics"))
        return SerLoggingCallsStatistics(nr_filtered_records, total_nr_records_in_dataset, calculated_features_statistics, experiments_results_statistics)

    def to_dict(self) -> dict:
        result: dict = {}
        result["nrFilteredRecords"] = from_int(self.nr_filtered_records)
        result["totalNrRecordsInDataset"] = from_int(self.total_nr_records_in_dataset)
        result["calculatedFeaturesStatistics"] = from_union([lambda x: from_list(lambda x: to_class(SerCalculatedFeatureStatistics, x), x), from_none], self.calculated_features_statistics)
        result["experimentsResultsStatistics"] = from_union([lambda x: from_list(lambda x: to_class(SerExperimentResultsStatistics, x), x), from_none], self.experiments_results_statistics)
        return result


@dataclass
class QLoggingCallStatistics:
    arguments: Optional[Dict[str, Any]] = None
    q_logging_call_statistics_return: Optional[SerLoggingCallsStatistics] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QLoggingCallStatistics':
        assert isinstance(obj, dict)
        arguments = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("arguments"))
        q_logging_call_statistics_return = from_union([SerLoggingCallsStatistics.from_dict, from_none], obj.get("return"))
        return QLoggingCallStatistics(arguments, q_logging_call_statistics_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: to_class(SerLoggingCallsStatistics, x), from_none], self.q_logging_call_statistics_return)
        return result


@dataclass
class QLoggingCalls:
    arguments: Optional[Dict[str, Any]] = None
    q_logging_calls_return: Optional[List[SerLoggingCall]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QLoggingCalls':
        assert isinstance(obj, dict)
        arguments = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("arguments"))
        q_logging_calls_return = from_union([lambda x: from_list(SerLoggingCall.from_dict, x), from_none], obj.get("return"))
        return QLoggingCalls(arguments, q_logging_calls_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: from_list(lambda x: to_class(SerLoggingCall, x), x), from_none], self.q_logging_calls_return)
        return result


@dataclass
class QMethodArguments:
    id: Any

    @staticmethod
    def from_dict(obj: Any) -> 'QMethodArguments':
        assert isinstance(obj, dict)
        id = obj.get("id")
        return QMethodArguments(id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = self.id
        return result


@dataclass
class SerParameter:
    name: str
    type_name: str

    @staticmethod
    def from_dict(obj: Any) -> 'SerParameter':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        type_name = from_str(obj.get("typeName"))
        return SerParameter(name, type_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["typeName"] = from_str(self.type_name)
        return result


@dataclass
class SerMethod:
    code: SerCodeSnippet
    code_filepath: str
    dataset_name: SerDatasetName
    declaration: SerCodeSnippet
    hash: str
    name: str
    return_type_name: str
    class_name: Optional[str] = None
    parameters: Optional[List[SerParameter]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerMethod':
        assert isinstance(obj, dict)
        code = SerCodeSnippet.from_dict(obj.get("code"))
        code_filepath = from_str(obj.get("codeFilepath"))
        dataset_name = SerDatasetName.from_dict(obj.get("datasetName"))
        declaration = SerCodeSnippet.from_dict(obj.get("declaration"))
        hash = from_str(obj.get("hash"))
        name = from_str(obj.get("name"))
        return_type_name = from_str(obj.get("returnTypeName"))
        class_name = from_union([from_str, from_none], obj.get("className"))
        parameters = from_union([lambda x: from_list(SerParameter.from_dict, x), from_none], obj.get("parameters"))
        return SerMethod(code, code_filepath, dataset_name, declaration, hash, name, return_type_name, class_name, parameters)

    def to_dict(self) -> dict:
        result: dict = {}
        result["code"] = to_class(SerCodeSnippet, self.code)
        result["codeFilepath"] = from_str(self.code_filepath)
        result["datasetName"] = to_class(SerDatasetName, self.dataset_name)
        result["declaration"] = to_class(SerCodeSnippet, self.declaration)
        result["hash"] = from_str(self.hash)
        result["name"] = from_str(self.name)
        result["returnTypeName"] = from_str(self.return_type_name)
        result["className"] = from_union([from_str, from_none], self.class_name)
        result["parameters"] = from_union([lambda x: from_list(lambda x: to_class(SerParameter, x), x), from_none], self.parameters)
        return result


@dataclass
class QMethod:
    arguments: Optional[QMethodArguments] = None
    q_method_return: Optional[SerMethod] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QMethod':
        assert isinstance(obj, dict)
        arguments = from_union([QMethodArguments.from_dict, from_none], obj.get("arguments"))
        q_method_return = from_union([SerMethod.from_dict, from_none], obj.get("return"))
        return QMethod(arguments, q_method_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: to_class(QMethodArguments, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: to_class(SerMethod, x), from_none], self.q_method_return)
        return result


@dataclass
class QMethodASTArguments:
    id: Any

    @staticmethod
    def from_dict(obj: Any) -> 'QMethodASTArguments':
        assert isinstance(obj, dict)
        id = obj.get("id")
        return QMethodASTArguments(id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = self.id
        return result


class SerASTNodeType(Enum):
    ARRAY_ACCESS_EXPR = "ArrayAccessExpr"
    ARRAY_BRACKET_PAIR = "ArrayBracketPair"
    ARRAY_CREATION_EXPR = "ArrayCreationExpr"
    ARRAY_CREATION_LEVEL = "ArrayCreationLevel"
    ARRAY_INITIALIZER_EXPR = "ArrayInitializerExpr"
    ARRAY_TYPE = "ArrayType"
    ASSERT_STMT = "AssertStmt"
    ASSIGN_EXPR_AND = "AssignExpr_AND"
    ASSIGN_EXPR_ASSIGN = "AssignExpr_ASSIGN"
    ASSIGN_EXPR_BINARY_AND = "AssignExpr_BINARY_AND"
    ASSIGN_EXPR_BINARY_OR = "AssignExpr_BINARY_OR"
    ASSIGN_EXPR_DIVIDE = "AssignExpr_DIVIDE"
    ASSIGN_EXPR_LEFT_SHIFT = "AssignExpr_LEFT_SHIFT"
    ASSIGN_EXPR_MINUS = "AssignExpr_MINUS"
    ASSIGN_EXPR_MULTIPLY = "AssignExpr_MULTIPLY"
    ASSIGN_EXPR_OR = "AssignExpr_OR"
    ASSIGN_EXPR_PLUS = "AssignExpr_PLUS"
    ASSIGN_EXPR_REM = "AssignExpr_REM"
    ASSIGN_EXPR_REMAINDER = "AssignExpr_REMAINDER"
    ASSIGN_EXPR_SIGNED_RIGHT_SHIFT = "AssignExpr_SIGNED_RIGHT_SHIFT"
    ASSIGN_EXPR_SLASH = "AssignExpr_SLASH"
    ASSIGN_EXPR_STAR = "AssignExpr_STAR"
    ASSIGN_EXPR_UNSIGNED_RIGHT_SHIFT = "AssignExpr_UNSIGNED_RIGHT_SHIFT"
    ASSIGN_EXPR_XOR = "AssignExpr_XOR"
    BINARY_EXPR_AND = "BinaryExpr_AND"
    BINARY_EXPR_BINARY_AND = "BinaryExpr_BINARY_AND"
    BINARY_EXPR_BINARY_OR = "BinaryExpr_BINARY_OR"
    BINARY_EXPR_DIVIDE = "BinaryExpr_DIVIDE"
    BINARY_EXPR_EQUALS = "BinaryExpr_EQUALS"
    BINARY_EXPR_GREATER = "BinaryExpr_GREATER"
    BINARY_EXPR_GREATER_EQUALS = "BinaryExpr_GREATER_EQUALS"
    BINARY_EXPR_LEFT_SHIFT = "BinaryExpr_LEFT_SHIFT"
    BINARY_EXPR_LESS = "BinaryExpr_LESS"
    BINARY_EXPR_LESS_EQUALS = "BinaryExpr_LESS_EQUALS"
    BINARY_EXPR_MINUS = "BinaryExpr_MINUS"
    BINARY_EXPR_MULTIPLY = "BinaryExpr_MULTIPLY"
    BINARY_EXPR_NOT_EQUALS = "BinaryExpr_NOT_EQUALS"
    BINARY_EXPR_OR = "BinaryExpr_OR"
    BINARY_EXPR_PLUS = "BinaryExpr_PLUS"
    BINARY_EXPR_REMAINDER = "BinaryExpr_REMAINDER"
    BINARY_EXPR_SIGNED_RIGHT_SHIFT = "BinaryExpr_SIGNED_RIGHT_SHIFT"
    BINARY_EXPR_TIMES = "BinaryExpr_TIMES"
    BINARY_EXPR_UNSIGNED_RIGHT_SHIFT = "BinaryExpr_UNSIGNED_RIGHT_SHIFT"
    BINARY_EXPR_XOR = "BinaryExpr_XOR"
    BLOCK_COMMENT = "BlockComment"
    BLOCK_STMT = "BlockStmt"
    BOOLEAN_LITERAL_EXPR = "BooleanLiteralExpr"
    BREAK_STMT = "BreakStmt"
    CAST_EXPR = "CastExpr"
    CATCH_CLAUSE = "CatchClause"
    CHAR_LITERAL_EXPR = "CharLiteralExpr"
    CLASS_EXPR = "ClassExpr"
    CLASS_OR_INTERFACE_DECLARATION = "ClassOrInterfaceDeclaration"
    CLASS_OR_INTERFACE_TYPE = "ClassOrInterfaceType"
    CONDITIONAL_EXPR = "ConditionalExpr"
    CONSTRUCTOR_DECLARATION = "ConstructorDeclaration"
    CONTINUE_STMT = "ContinueStmt"
    DOUBLE_LITERAL_EXPR = "DoubleLiteralExpr"
    DO_STMT = "DoStmt"
    EMPTY_MEMBER_DECLARATION = "EmptyMemberDeclaration"
    EMPTY_STMT = "EmptyStmt"
    ENCLOSED_EXPR = "EnclosedExpr"
    EOS = "EOS"
    EXPLICIT_CONSTRUCTOR_INVOCATION_STMT = "ExplicitConstructorInvocationStmt"
    EXPRESSION_STMT = "ExpressionStmt"
    FAKE_NODE = "FakeNode"
    FIELD_ACCESS_EXPR = "FieldAccessExpr"
    FIELD_DECLARATION = "FieldDeclaration"
    FOR_EACH_STMT = "ForEachStmt"
    FOR_STMT = "ForStmt"
    GENERIC_CLASS = "GenericClass"
    IF_STMT = "IfStmt"
    INITIALIZER_DECLARATION = "InitializerDeclaration"
    INSTANCE_OF_EXPR = "InstanceOfExpr"
    INTEGER_LITERAL_EXPR = "IntegerLiteralExpr"
    INTEGER_LITERAL_MIN_VALUE_EXPR = "IntegerLiteralMinValueExpr"
    JAVADOC_COMMENT = "JavadocComment"
    LABELED_STMT = "LabeledStmt"
    LAMBDA_EXPR = "LambdaExpr"
    LINE_COMMENT = "LineComment"
    LOCAL_CLASS_DECLARATION_STMT = "LocalClassDeclarationStmt"
    LONG_LITERAL_EXPR = "LongLiteralExpr"
    MARKER_ANNOTATION_EXPR = "MarkerAnnotationExpr"
    MEMBER_VALUE_PAIR = "MemberValuePair"
    METHOD_CALL_EXPR = "MethodCallExpr"
    METHOD_DECLARATION = "MethodDeclaration"
    METHOD_REFERENCE_EXPR = "MethodReferenceExpr"
    MODIFIER = "Modifier"
    NAME = "Name"
    NAME_EXPR = "NameExpr"
    NORMAL_ANNOTATION_EXPR = "NormalAnnotationExpr"
    NULL_LITERAL_EXPR = "NullLiteralExpr"
    OBJECT_CREATION_EXPR = "ObjectCreationExpr"
    PARAMETER = "Parameter"
    PRIMITIVE_TYPE = "PrimitiveType"
    QUALIFIED_NAME_EXPR = "QualifiedNameExpr"
    RETURN_STMT = "ReturnStmt"
    SIMPLE_NAME = "SimpleName"
    SINGLE_MEMBER_ANNOTATION_EXPR = "SingleMemberAnnotationExpr"
    STATIC_CALL = "StaticCall"
    STRING_LITERAL_EXPR = "StringLiteralExpr"
    SUPER_EXPR = "SuperExpr"
    SWITCH_ENTRY = "SwitchEntry"
    SWITCH_ENTRY_STMT = "SwitchEntryStmt"
    SWITCH_STMT = "SwitchStmt"
    SYNCHRONIZED_STMT = "SynchronizedStmt"
    THIS_EXPR = "ThisExpr"
    THROW_STMT = "ThrowStmt"
    TRY_STMT = "TryStmt"
    TYPE_DECLARATION_STMT = "TypeDeclarationStmt"
    TYPE_EXPR = "TypeExpr"
    TYPE_PARAMETER = "TypeParameter"
    UNARY_EXPR_BITWISE_COMPLEMENT = "UnaryExpr_BITWISE_COMPLEMENT"
    UNARY_EXPR_INVERSE = "UnaryExpr_INVERSE"
    UNARY_EXPR_LOGICAL_COMPLEMENT = "UnaryExpr_LOGICAL_COMPLEMENT"
    UNARY_EXPR_MINUS = "UnaryExpr_MINUS"
    UNARY_EXPR_NEGATIVE = "UnaryExpr_NEGATIVE"
    UNARY_EXPR_NOT = "UnaryExpr_NOT"
    UNARY_EXPR_PLUS = "UnaryExpr_PLUS"
    UNARY_EXPR_POSITIVE = "UnaryExpr_POSITIVE"
    UNARY_EXPR_POSTFIX_DECREMENT = "UnaryExpr_POSTFIX_DECREMENT"
    UNARY_EXPR_POSTFIX_INCREMENT = "UnaryExpr_POSTFIX_INCREMENT"
    UNARY_EXPR_POS_DECREMENT = "UnaryExpr_POS_DECREMENT"
    UNARY_EXPR_POS_INCREMENT = "UnaryExpr_POS_INCREMENT"
    UNARY_EXPR_PREFIX_DECREMENT = "UnaryExpr_PREFIX_DECREMENT"
    UNARY_EXPR_PREFIX_INCREMENT = "UnaryExpr_PREFIX_INCREMENT"
    UNARY_EXPR_PRE_DECREMENT = "UnaryExpr_PRE_DECREMENT"
    UNARY_EXPR_PRE_INCREMENT = "UnaryExpr_PRE_INCREMENT"
    UNION_TYPE = "UnionType"
    UNKNOWN_TYPE = "UnknownType"
    VARIABLE_DECLARATION_EXPR = "VariableDeclarationExpr"
    VARIABLE_DECLARATOR = "VariableDeclarator"
    VARIABLE_DECLARATOR_ID = "VariableDeclaratorId"
    VOID_TYPE = "VoidType"
    WHILE_STMT = "WhileStmt"
    WILDCARD_TYPE = "WildcardType"


@dataclass
class SerASTNode:
    idx: int
    type: SerASTNodeType
    child_place_at_parent: Optional[int] = None
    children_idxs: Optional[List[int]] = None
    identifier: Optional[str] = None
    literal_expr: Optional[str] = None
    modifier: Optional[str] = None
    name: Optional[str] = None
    parent_node_idx: Optional[int] = None
    type_name: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerASTNode':
        assert isinstance(obj, dict)
        idx = from_int(obj.get("idx"))
        type = SerASTNodeType(obj.get("type"))
        child_place_at_parent = from_union([from_int, from_none], obj.get("childPlaceAtParent"))
        children_idxs = from_union([lambda x: from_list(from_int, x), from_none], obj.get("childrenIdxs"))
        identifier = from_union([from_str, from_none], obj.get("identifier"))
        literal_expr = from_union([from_str, from_none], obj.get("literalExpr"))
        modifier = from_union([from_str, from_none], obj.get("modifier"))
        name = from_union([from_str, from_none], obj.get("name"))
        parent_node_idx = from_union([from_int, from_none], obj.get("parentNodeIdx"))
        type_name = from_union([from_str, from_none], obj.get("typeName"))
        return SerASTNode(idx, type, child_place_at_parent, children_idxs, identifier, literal_expr, modifier, name, parent_node_idx, type_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["idx"] = from_int(self.idx)
        result["type"] = to_enum(SerASTNodeType, self.type)
        result["childPlaceAtParent"] = from_union([from_int, from_none], self.child_place_at_parent)
        result["childrenIdxs"] = from_union([lambda x: from_list(from_int, x), from_none], self.children_idxs)
        result["identifier"] = from_union([from_str, from_none], self.identifier)
        result["literalExpr"] = from_union([from_str, from_none], self.literal_expr)
        result["modifier"] = from_union([from_str, from_none], self.modifier)
        result["name"] = from_union([from_str, from_none], self.name)
        result["parentNodeIdx"] = from_union([from_int, from_none], self.parent_node_idx)
        result["typeName"] = from_union([from_str, from_none], self.type_name)
        return result


@dataclass
class SerMethodAST:
    method_hash: str
    root_node_idx: int
    nodes: Optional[List[SerASTNode]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerMethodAST':
        assert isinstance(obj, dict)
        method_hash = from_str(obj.get("methodHash"))
        root_node_idx = from_int(obj.get("rootNodeIdx"))
        nodes = from_union([lambda x: from_list(SerASTNode.from_dict, x), from_none], obj.get("nodes"))
        return SerMethodAST(method_hash, root_node_idx, nodes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["methodHash"] = from_str(self.method_hash)
        result["rootNodeIdx"] = from_int(self.root_node_idx)
        result["nodes"] = from_union([lambda x: from_list(lambda x: to_class(SerASTNode, x), x), from_none], self.nodes)
        return result


@dataclass
class QMethodAST:
    arguments: Optional[QMethodASTArguments] = None
    q_method_ast_return: Optional[SerMethodAST] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QMethodAST':
        assert isinstance(obj, dict)
        arguments = from_union([QMethodASTArguments.from_dict, from_none], obj.get("arguments"))
        q_method_ast_return = from_union([SerMethodAST.from_dict, from_none], obj.get("return"))
        return QMethodAST(arguments, q_method_ast_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: to_class(QMethodASTArguments, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: to_class(SerMethodAST, x), from_none], self.q_method_ast_return)
        return result


@dataclass
class QMethodPDGArguments:
    id: Any

    @staticmethod
    def from_dict(obj: Any) -> 'QMethodPDGArguments':
        assert isinstance(obj, dict)
        id = obj.get("id")
        return QMethodPDGArguments(id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = self.id
        return result


class SerControlScopeType(Enum):
    ASSERT = "Assert"
    BREAK = "Break"
    CONTINUE = "Continue"
    EMPTY_INFINITE_LOOP_BODY = "EmptyInfiniteLoopBody"
    IF = "If"
    IF_CONDITION = "IfCondition"
    IF_ELSE_BODY = "IfElseBody"
    IF_THEN_BODY = "IfThenBody"
    LOOP = "Loop"
    LOOP_BODY = "LoopBody"
    LOOP_BREAK = "LoopBreak"
    LOOP_CONDITION = "LoopCondition"
    LOOP_CONTINUE = "LoopContinue"
    LOOP_INIT = "LoopInit"
    LOOP_UPDATE = "LoopUpdate"
    LOOP_UPDATE_AND_CONDITION = "LoopUpdateAndCondition"
    LOOP_WITH_INIT = "LoopWithInit"
    METHOD = "Method"
    METHOD_BODY = "MethodBody"
    METHOD_ENTRY = "MethodEntry"
    METHOD_EXIT = "MethodExit"
    METHOD_RETURN = "MethodReturn"
    METHOD_YIELD = "MethodYield"
    RETURN = "Return"
    SWITCH = "Switch"
    SWITCH_ENTRY = "SwitchEntry"
    SYNC = "Sync"
    SYNC_BODY = "SyncBody"
    THROW = "Throw"
    TRY = "Try"
    TRY_BODY = "TryBody"
    TRY_CATCH = "TryCatch"
    TRY_FINALLY = "TryFinally"
    TRY_RESOURCES = "TryResources"
    YIELD = "Yield"


@dataclass
class SerControlScope:
    idx: int
    type: SerControlScopeType
    ast_node_idx: Optional[int] = None
    entry_point_pdg_node_idx: Optional[int] = None
    exit_points_pdg_node_idx: Optional[List[int]] = None
    own_symbols_scope_idx: Optional[int] = None
    parent_control_scope_idx: Optional[int] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerControlScope':
        assert isinstance(obj, dict)
        idx = from_int(obj.get("idx"))
        type = SerControlScopeType(obj.get("type"))
        ast_node_idx = from_union([from_int, from_none], obj.get("astNodeIdx"))
        entry_point_pdg_node_idx = from_union([from_int, from_none], obj.get("entryPointPDGNodeIdx"))
        exit_points_pdg_node_idx = from_union([lambda x: from_list(from_int, x), from_none], obj.get("exitPointsPDGNodeIdx"))
        own_symbols_scope_idx = from_union([from_int, from_none], obj.get("ownSymbolsScopeIdx"))
        parent_control_scope_idx = from_union([from_int, from_none], obj.get("parentControlScopeIdx"))
        return SerControlScope(idx, type, ast_node_idx, entry_point_pdg_node_idx, exit_points_pdg_node_idx, own_symbols_scope_idx, parent_control_scope_idx)

    def to_dict(self) -> dict:
        result: dict = {}
        result["idx"] = from_int(self.idx)
        result["type"] = to_enum(SerControlScopeType, self.type)
        result["astNodeIdx"] = from_union([from_int, from_none], self.ast_node_idx)
        result["entryPointPDGNodeIdx"] = from_union([from_int, from_none], self.entry_point_pdg_node_idx)
        result["exitPointsPDGNodeIdx"] = from_union([lambda x: from_list(from_int, x), from_none], self.exit_points_pdg_node_idx)
        result["ownSymbolsScopeIdx"] = from_union([from_int, from_none], self.own_symbols_scope_idx)
        result["parentControlScopeIdx"] = from_union([from_int, from_none], self.parent_control_scope_idx)
        return result


class SerControlFlowEdgeType(Enum):
    ASSERT_FAILURE = "AssertFailure"
    ASSERT_SUCCEED = "AssertSucceed"
    BREAK = "Break"
    CONTINUE = "Continue"
    EXPLICIT_THROW = "ExplicitThrow"
    FORWARD = "Forward"
    IMPLICIT_THROW = "ImplicitThrow"
    ON_FALSE = "OnFalse"
    ON_FALSE_LOOP_OUT = "OnFalseLoopOut"
    ON_TRUE = "OnTrue"
    ON_TRUE_LOOP_BACK = "OnTrueLoopBack"
    RETURN = "Return"
    YIELD = "Yield"
    YIELD_BACK = "YieldBack"


@dataclass
class SerPDGControlFlowEdge:
    pgd_node_idx: int
    type: SerControlFlowEdgeType

    @staticmethod
    def from_dict(obj: Any) -> 'SerPDGControlFlowEdge':
        assert isinstance(obj, dict)
        pgd_node_idx = from_int(obj.get("PGDNodeIdx"))
        type = SerControlFlowEdgeType(obj.get("type"))
        return SerPDGControlFlowEdge(pgd_node_idx, type)

    def to_dict(self) -> dict:
        result: dict = {}
        result["PGDNodeIdx"] = from_int(self.pgd_node_idx)
        result["type"] = to_enum(SerControlFlowEdgeType, self.type)
        return result


class SerPDGNodeControlKind(Enum):
    ASSERT = "Assert"
    BREAK = "Break"
    CATCH_CLAUSE_ENTRY = "CatchClauseEntry"
    CLASS_THIS_OBJECT_BINDING_TO_METHOD = "ClassThisObjectBindingToMethod"
    CONDITION = "Condition"
    CONTINUE = "Continue"
    ITERATOR_INIT = "IteratorInit"
    ITERATOR_NEXT = "IteratorNext"
    METHOD_ENTRY = "MethodEntry"
    METHOD_EXIT = "MethodExit"
    RETURN = "Return"
    SIMPLE_STATEMENT = "SimpleStatement"
    SYNC = "Sync"
    THROW = "Throw"
    WITH = "With"
    YIELD = "Yield"


@dataclass
class SerSymbolRef:
    identifier_idx: int
    symbol_idx: int
    symbol_name: str

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolRef':
        assert isinstance(obj, dict)
        identifier_idx = from_int(obj.get("identifierIdx"))
        symbol_idx = from_int(obj.get("symbolIdx"))
        symbol_name = from_str(obj.get("symbolName"))
        return SerSymbolRef(identifier_idx, symbol_idx, symbol_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["identifierIdx"] = from_int(self.identifier_idx)
        result["symbolIdx"] = from_int(self.symbol_idx)
        result["symbolName"] = from_str(self.symbol_name)
        return result


@dataclass
class SerPDGDataDependencyEdge:
    pgd_node_idx: int
    symbols: Optional[List[SerSymbolRef]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerPDGDataDependencyEdge':
        assert isinstance(obj, dict)
        pgd_node_idx = from_int(obj.get("PGDNodeIdx"))
        symbols = from_union([lambda x: from_list(SerSymbolRef.from_dict, x), from_none], obj.get("symbols"))
        return SerPDGDataDependencyEdge(pgd_node_idx, symbols)

    def to_dict(self) -> dict:
        result: dict = {}
        result["PGDNodeIdx"] = from_int(self.pgd_node_idx)
        result["symbols"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbolRef, x), x), from_none], self.symbols)
        return result


@dataclass
class SerMayMustSymbolRefsList:
    may: Optional[List[SerSymbolRef]] = None
    must: Optional[List[SerSymbolRef]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerMayMustSymbolRefsList':
        assert isinstance(obj, dict)
        may = from_union([lambda x: from_list(SerSymbolRef.from_dict, x), from_none], obj.get("may"))
        must = from_union([lambda x: from_list(SerSymbolRef.from_dict, x), from_none], obj.get("must"))
        return SerMayMustSymbolRefsList(may, must)

    def to_dict(self) -> dict:
        result: dict = {}
        result["may"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbolRef, x), x), from_none], self.may)
        result["must"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbolRef, x), x), from_none], self.must)
        return result


@dataclass
class SerSymbolsUseDefMut:
    ser_symbols_use_def_mut_def: SerMayMustSymbolRefsList
    mut: SerMayMustSymbolRefsList
    use: SerMayMustSymbolRefsList

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolsUseDefMut':
        assert isinstance(obj, dict)
        ser_symbols_use_def_mut_def = SerMayMustSymbolRefsList.from_dict(obj.get("def"))
        mut = SerMayMustSymbolRefsList.from_dict(obj.get("mut"))
        use = SerMayMustSymbolRefsList.from_dict(obj.get("use"))
        return SerSymbolsUseDefMut(ser_symbols_use_def_mut_def, mut, use)

    def to_dict(self) -> dict:
        result: dict = {}
        result["def"] = to_class(SerMayMustSymbolRefsList, self.ser_symbols_use_def_mut_def)
        result["mut"] = to_class(SerMayMustSymbolRefsList, self.mut)
        result["use"] = to_class(SerMayMustSymbolRefsList, self.use)
        return result


@dataclass
class SerPDGNode:
    control_kind: SerPDGNodeControlKind
    has_expression: bool
    idx: int
    ast_node_idx: Optional[int] = None
    belongs_to_control_scopes_idxs: Optional[List[int]] = None
    called_functions_names: Optional[List[str]] = None
    code: Optional[SerCodeSnippet] = None
    control_flow_in_edges: Optional[List[SerPDGControlFlowEdge]] = None
    control_flow_out_edges: Optional[List[SerPDGControlFlowEdge]] = None
    data_dependency_in_edges: Optional[List[SerPDGDataDependencyEdge]] = None
    data_dependency_out_edges: Optional[List[SerPDGDataDependencyEdge]] = None
    symbols_use_def_mut: Optional[SerSymbolsUseDefMut] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerPDGNode':
        assert isinstance(obj, dict)
        control_kind = SerPDGNodeControlKind(obj.get("controlKind"))
        has_expression = from_bool(obj.get("hasExpression"))
        idx = from_int(obj.get("idx"))
        ast_node_idx = from_union([from_int, from_none], obj.get("astNodeIdx"))
        belongs_to_control_scopes_idxs = from_union([lambda x: from_list(from_int, x), from_none], obj.get("belongsToControlScopesIdxs"))
        called_functions_names = from_union([lambda x: from_list(from_str, x), from_none], obj.get("calledFunctionsNames"))
        code = from_union([SerCodeSnippet.from_dict, from_none], obj.get("code"))
        control_flow_in_edges = from_union([lambda x: from_list(SerPDGControlFlowEdge.from_dict, x), from_none], obj.get("controlFlowInEdges"))
        control_flow_out_edges = from_union([lambda x: from_list(SerPDGControlFlowEdge.from_dict, x), from_none], obj.get("controlFlowOutEdges"))
        data_dependency_in_edges = from_union([lambda x: from_list(SerPDGDataDependencyEdge.from_dict, x), from_none], obj.get("dataDependencyInEdges"))
        data_dependency_out_edges = from_union([lambda x: from_list(SerPDGDataDependencyEdge.from_dict, x), from_none], obj.get("dataDependencyOutEdges"))
        symbols_use_def_mut = from_union([SerSymbolsUseDefMut.from_dict, from_none], obj.get("symbolsUseDefMut"))
        return SerPDGNode(control_kind, has_expression, idx, ast_node_idx, belongs_to_control_scopes_idxs, called_functions_names, code, control_flow_in_edges, control_flow_out_edges, data_dependency_in_edges, data_dependency_out_edges, symbols_use_def_mut)

    def to_dict(self) -> dict:
        result: dict = {}
        result["controlKind"] = to_enum(SerPDGNodeControlKind, self.control_kind)
        result["hasExpression"] = from_bool(self.has_expression)
        result["idx"] = from_int(self.idx)
        result["astNodeIdx"] = from_union([from_int, from_none], self.ast_node_idx)
        result["belongsToControlScopesIdxs"] = from_union([lambda x: from_list(from_int, x), from_none], self.belongs_to_control_scopes_idxs)
        result["calledFunctionsNames"] = from_union([lambda x: from_list(from_str, x), from_none], self.called_functions_names)
        result["code"] = from_union([lambda x: to_class(SerCodeSnippet, x), from_none], self.code)
        result["controlFlowInEdges"] = from_union([lambda x: from_list(lambda x: to_class(SerPDGControlFlowEdge, x), x), from_none], self.control_flow_in_edges)
        result["controlFlowOutEdges"] = from_union([lambda x: from_list(lambda x: to_class(SerPDGControlFlowEdge, x), x), from_none], self.control_flow_out_edges)
        result["dataDependencyInEdges"] = from_union([lambda x: from_list(lambda x: to_class(SerPDGDataDependencyEdge, x), x), from_none], self.data_dependency_in_edges)
        result["dataDependencyOutEdges"] = from_union([lambda x: from_list(lambda x: to_class(SerPDGDataDependencyEdge, x), x), from_none], self.data_dependency_out_edges)
        result["symbolsUseDefMut"] = from_union([lambda x: to_class(SerSymbolsUseDefMut, x), from_none], self.symbols_use_def_mut)
        return result


class SerSymbolDeclarationKind(Enum):
    FIELD_OF_THIS = "FieldOfThis"
    LOCAL_VARIABLE = "LocalVariable"
    METHOD = "Method"
    PARAMETER = "Parameter"


@dataclass
class SerSymbol:
    contained_in_symbols_scope_idx: int
    declaration_kind: SerSymbolDeclarationKind
    decl_site_pdg_node_idx: int
    identifier_idx: int
    symbol_name: str
    type_name: str
    declaration_ast_node_idx: Optional[int] = None
    def_sites_pdg_node_ids: Optional[List[int]] = None
    use_sites_pdg_node_ids: Optional[List[int]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbol':
        assert isinstance(obj, dict)
        contained_in_symbols_scope_idx = from_int(obj.get("containedInSymbolsScopeIdx"))
        declaration_kind = SerSymbolDeclarationKind(obj.get("declarationKind"))
        decl_site_pdg_node_idx = from_int(obj.get("declSitePDGNodeIdx"))
        identifier_idx = from_int(obj.get("identifierIdx"))
        symbol_name = from_str(obj.get("symbolName"))
        type_name = from_str(obj.get("typeName"))
        declaration_ast_node_idx = from_union([from_int, from_none], obj.get("declarationASTNodeIdx"))
        def_sites_pdg_node_ids = from_union([lambda x: from_list(from_int, x), from_none], obj.get("defSitesPDGNodeIds"))
        use_sites_pdg_node_ids = from_union([lambda x: from_list(from_int, x), from_none], obj.get("useSitesPDGNodeIds"))
        return SerSymbol(contained_in_symbols_scope_idx, declaration_kind, decl_site_pdg_node_idx, identifier_idx, symbol_name, type_name, declaration_ast_node_idx, def_sites_pdg_node_ids, use_sites_pdg_node_ids)

    def to_dict(self) -> dict:
        result: dict = {}
        result["containedInSymbolsScopeIdx"] = from_int(self.contained_in_symbols_scope_idx)
        result["declarationKind"] = to_enum(SerSymbolDeclarationKind, self.declaration_kind)
        result["declSitePDGNodeIdx"] = from_int(self.decl_site_pdg_node_idx)
        result["identifierIdx"] = from_int(self.identifier_idx)
        result["symbolName"] = from_str(self.symbol_name)
        result["typeName"] = from_str(self.type_name)
        result["declarationASTNodeIdx"] = from_union([from_int, from_none], self.declaration_ast_node_idx)
        result["defSitesPDGNodeIds"] = from_union([lambda x: from_list(from_int, x), from_none], self.def_sites_pdg_node_ids)
        result["useSitesPDGNodeIds"] = from_union([lambda x: from_list(from_int, x), from_none], self.use_sites_pdg_node_ids)
        return result


@dataclass
class SerSymbolsScope:
    idx: int
    parent_symbols_scope_idx: Optional[int] = None
    symbols: Optional[List[SerSymbol]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerSymbolsScope':
        assert isinstance(obj, dict)
        idx = from_int(obj.get("idx"))
        parent_symbols_scope_idx = from_union([from_int, from_none], obj.get("parentSymbolsScopeIdx"))
        symbols = from_union([lambda x: from_list(SerSymbol.from_dict, x), from_none], obj.get("symbols"))
        return SerSymbolsScope(idx, parent_symbols_scope_idx, symbols)

    def to_dict(self) -> dict:
        result: dict = {}
        result["idx"] = from_int(self.idx)
        result["parentSymbolsScopeIdx"] = from_union([from_int, from_none], self.parent_symbols_scope_idx)
        result["symbols"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbol, x), x), from_none], self.symbols)
        return result


@dataclass
class SerMethodPDG:
    entry_pdg_node_idx: int
    exit_pdg_node_idx: int
    method_hash: str
    control_scopes: Optional[List[SerControlScope]] = None
    identifier_by_idx: Optional[List[str]] = None
    pdg_nodes: Optional[List[SerPDGNode]] = None
    sub_identifiers_by_idx: Optional[List[List[str]]] = None
    symbols_scopes: Optional[List[SerSymbolsScope]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SerMethodPDG':
        assert isinstance(obj, dict)
        entry_pdg_node_idx = from_int(obj.get("entryPDGNodeIdx"))
        exit_pdg_node_idx = from_int(obj.get("exitPDGNodeIdx"))
        method_hash = from_str(obj.get("methodHash"))
        control_scopes = from_union([lambda x: from_list(SerControlScope.from_dict, x), from_none], obj.get("controlScopes"))
        identifier_by_idx = from_union([lambda x: from_list(from_str, x), from_none], obj.get("identifierByIdx"))
        pdg_nodes = from_union([lambda x: from_list(SerPDGNode.from_dict, x), from_none], obj.get("pdgNodes"))
        sub_identifiers_by_idx = from_union([lambda x: from_list(lambda x: from_list(from_str, x), x), from_none], obj.get("subIdentifiersByIdx"))
        symbols_scopes = from_union([lambda x: from_list(SerSymbolsScope.from_dict, x), from_none], obj.get("symbolsScopes"))
        return SerMethodPDG(entry_pdg_node_idx, exit_pdg_node_idx, method_hash, control_scopes, identifier_by_idx, pdg_nodes, sub_identifiers_by_idx, symbols_scopes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["entryPDGNodeIdx"] = from_int(self.entry_pdg_node_idx)
        result["exitPDGNodeIdx"] = from_int(self.exit_pdg_node_idx)
        result["methodHash"] = from_str(self.method_hash)
        result["controlScopes"] = from_union([lambda x: from_list(lambda x: to_class(SerControlScope, x), x), from_none], self.control_scopes)
        result["identifierByIdx"] = from_union([lambda x: from_list(from_str, x), from_none], self.identifier_by_idx)
        result["pdgNodes"] = from_union([lambda x: from_list(lambda x: to_class(SerPDGNode, x), x), from_none], self.pdg_nodes)
        result["subIdentifiersByIdx"] = from_union([lambda x: from_list(lambda x: from_list(from_str, x), x), from_none], self.sub_identifiers_by_idx)
        result["symbolsScopes"] = from_union([lambda x: from_list(lambda x: to_class(SerSymbolsScope, x), x), from_none], self.symbols_scopes)
        return result


@dataclass
class QMethodPDG:
    arguments: Optional[QMethodPDGArguments] = None
    q_method_pdg_return: Optional[SerMethodPDG] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QMethodPDG':
        assert isinstance(obj, dict)
        arguments = from_union([QMethodPDGArguments.from_dict, from_none], obj.get("arguments"))
        q_method_pdg_return = from_union([SerMethodPDG.from_dict, from_none], obj.get("return"))
        return QMethodPDG(arguments, q_method_pdg_return)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arguments"] = from_union([lambda x: to_class(QMethodPDGArguments, x), from_none], self.arguments)
        result["return"] = from_union([lambda x: to_class(SerMethodPDG, x), from_none], self.q_method_pdg_return)
        return result


@dataclass
class Query:
    q_dummy_get_filtes: Optional[QDummyGetFiltes] = None
    q_experiments: Optional[QExperiments] = None
    q_logging_call: Optional[QLoggingCall] = None
    q_logging_call_features: Optional[QLoggingCallFeatures] = None
    q_logging_calls: Optional[QLoggingCalls] = None
    q_logging_call_statistics: Optional[QLoggingCallStatistics] = None
    q_method: Optional[QMethod] = None
    q_method_ast: Optional[QMethodAST] = None
    q_method_pdg: Optional[QMethodPDG] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Query':
        assert isinstance(obj, dict)
        q_dummy_get_filtes = from_union([QDummyGetFiltes.from_dict, from_none], obj.get("qDummyGetFiltes"))
        q_experiments = from_union([QExperiments.from_dict, from_none], obj.get("qExperiments"))
        q_logging_call = from_union([QLoggingCall.from_dict, from_none], obj.get("qLoggingCall"))
        q_logging_call_features = from_union([QLoggingCallFeatures.from_dict, from_none], obj.get("qLoggingCallFeatures"))
        q_logging_calls = from_union([QLoggingCalls.from_dict, from_none], obj.get("qLoggingCalls"))
        q_logging_call_statistics = from_union([QLoggingCallStatistics.from_dict, from_none], obj.get("qLoggingCallStatistics"))
        q_method = from_union([QMethod.from_dict, from_none], obj.get("qMethod"))
        q_method_ast = from_union([QMethodAST.from_dict, from_none], obj.get("qMethodAST"))
        q_method_pdg = from_union([QMethodPDG.from_dict, from_none], obj.get("qMethodPDG"))
        return Query(q_dummy_get_filtes, q_experiments, q_logging_call, q_logging_call_features, q_logging_calls, q_logging_call_statistics, q_method, q_method_ast, q_method_pdg)

    def to_dict(self) -> dict:
        result: dict = {}
        result["qDummyGetFiltes"] = from_union([lambda x: to_class(QDummyGetFiltes, x), from_none], self.q_dummy_get_filtes)
        result["qExperiments"] = from_union([lambda x: to_class(QExperiments, x), from_none], self.q_experiments)
        result["qLoggingCall"] = from_union([lambda x: to_class(QLoggingCall, x), from_none], self.q_logging_call)
        result["qLoggingCallFeatures"] = from_union([lambda x: to_class(QLoggingCallFeatures, x), from_none], self.q_logging_call_features)
        result["qLoggingCalls"] = from_union([lambda x: to_class(QLoggingCalls, x), from_none], self.q_logging_calls)
        result["qLoggingCallStatistics"] = from_union([lambda x: to_class(QLoggingCallStatistics, x), from_none], self.q_logging_call_statistics)
        result["qMethod"] = from_union([lambda x: to_class(QMethod, x), from_none], self.q_method)
        result["qMethodAST"] = from_union([lambda x: to_class(QMethodAST, x), from_none], self.q_method_ast)
        result["qMethodPDG"] = from_union([lambda x: to_class(QMethodPDG, x), from_none], self.q_method_pdg)
        return result


@dataclass
class LoggingCallAPIClass:
    query: Optional[Query] = None

    @staticmethod
    def from_dict(obj: Any) -> 'LoggingCallAPIClass':
        assert isinstance(obj, dict)
        query = from_union([Query.from_dict, from_none], obj.get("Query"))
        return LoggingCallAPIClass(query)

    def to_dict(self) -> dict:
        result: dict = {}
        result["Query"] = from_union([lambda x: to_class(Query, x), from_none], self.query)
        return result


def logging_call_api_from_dict(s: Any) -> Union[List[Any], bool, LoggingCallAPIClass, float, int, None, str]:
    return from_union([from_none, from_float, from_int, from_bool, from_str, lambda x: from_list(lambda x: x, x), LoggingCallAPIClass.from_dict], s)


def logging_call_api_to_dict(x: Union[List[Any], bool, LoggingCallAPIClass, float, int, None, str]) -> Any:
    return from_union([from_none, to_float, from_int, from_bool, from_str, lambda x: from_list(lambda x: x, x), lambda x: to_class(LoggingCallAPIClass, x)], x)
