"""Regex builder tool for creating, testing, and explaining regular expressions.

This tool provides natural language to regex conversion, pattern testing,
and human-readable explanations of regex patterns.
"""

import re
from typing import Optional


class RegexBuilderTool:
    """Tool for building, testing, and explaining regular expressions."""

    def __init__(self):
        """Initialize the regex builder tool."""
        self._common_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "url": r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_+.~#?&/=]*",
            "phone": r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
            "phone_international": r"\+?[0-9]{1,4}[-.\s]?\(?[0-9]{1,4}\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}",
            "ip_address": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "ipv6": r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}",
            "date_iso": r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])",
            "date_us": r"(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/\d{4}",
            "date_eu": r"(?:0[1-9]|[12][0-9]|3[01])/(?:0[1-9]|1[0-2])/\d{4}",
            "time_24h": r"(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?",
            "time_12h": r"(?:0?[1-9]|1[0-2]):[0-5][0-9](?::[0-5][0-9])?\s*[AaPp][Mm]",
            "hex_color": r"#(?:[0-9a-fA-F]{3}){1,2}\b",
            "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "zip_code_us": r"\b\d{5}(?:-\d{4})?\b",
            "username": r"^[a-zA-Z][a-zA-Z0-9_-]{2,15}$",
            "password_strong": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",
            "slug": r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
            "uuid": r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "mac_address": r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}",
            "html_tag": r"<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>.*?</\1>|<[a-zA-Z][a-zA-Z0-9]*\b[^/>]*/>",
            "whitespace": r"\s+",
            "word": r"\b\w+\b",
            "integer": r"-?\d+",
            "float": r"-?\d+\.?\d*",
            "alphanumeric": r"^[a-zA-Z0-9]+$",
        }

    def build(self, description: str) -> dict:
        """Build a regex pattern from a natural language description.

        Converts natural language descriptions into regex patterns with
        explanations of each component.

        Args:
            description: Natural language description of what to match.
                Examples: "match email addresses", "find phone numbers",
                "extract URLs from text"

        Returns:
            dict with:
                - success: bool
                - pattern: The generated regex pattern
                - explanation: Human-readable explanation of the pattern
                - example_matches: Sample strings that would match

        Example:
            >>> tool = RegexBuilderTool()
            >>> result = tool.build("match email addresses")
            >>> result['pattern']
            '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'
        """
        if not description:
            return {
                "success": False,
                "error": "No description provided"
            }

        desc_lower = description.lower()

        # Map common descriptions to patterns
        pattern_map = {
            ("email", "e-mail", "mail address"): "email",
            ("url", "web address", "website", "link", "http"): "url",
            ("phone", "telephone", "mobile", "cell"): "phone",
            ("ip address", "ip addr", "ipv4"): "ip_address",
            ("ipv6",): "ipv6",
            ("date", "yyyy-mm-dd", "iso date"): "date_iso",
            ("us date", "mm/dd/yyyy", "american date"): "date_us",
            ("eu date", "dd/mm/yyyy", "european date"): "date_eu",
            ("time", "24 hour", "24h"): "time_24h",
            ("12 hour", "12h", "am pm", "am/pm"): "time_12h",
            ("hex color", "color code", "hex code", "#"): "hex_color",
            ("credit card", "card number", "cc number"): "credit_card",
            ("ssn", "social security"): "ssn",
            ("zip code", "postal code", "zip"): "zip_code_us",
            ("username", "user name"): "username",
            ("strong password", "secure password"): "password_strong",
            ("slug", "url slug"): "slug",
            ("uuid", "guid"): "uuid",
            ("mac address", "mac addr"): "mac_address",
            ("html tag", "html element"): "html_tag",
            ("whitespace", "spaces", "blank"): "whitespace",
            ("word", "words"): "word",
            ("integer", "whole number", "int"): "integer",
            ("float", "decimal", "number with decimal"): "float",
            ("alphanumeric", "letters and numbers"): "alphanumeric",
        }

        # Find matching pattern
        matched_key = None
        for keywords, pattern_key in pattern_map.items():
            if any(kw in desc_lower for kw in keywords):
                matched_key = pattern_key
                break

        if matched_key:
            pattern = self._common_patterns[matched_key]
            explanation = self._explain_pattern(pattern)
            examples = self._generate_examples(matched_key)
            return {
                "success": True,
                "pattern": pattern,
                "explanation": explanation,
                "example_matches": examples,
                "pattern_name": matched_key
            }

        # Try to build a basic pattern from description
        basic_pattern = self._build_basic_pattern(description)
        if basic_pattern:
            return {
                "success": True,
                "pattern": basic_pattern["pattern"],
                "explanation": basic_pattern["explanation"],
                "example_matches": basic_pattern.get("examples", []),
                "note": "Generated basic pattern - may need refinement"
            }

        return {
            "success": False,
            "error": f"Could not generate pattern for: {description}",
            "suggestion": "Try using common_patterns() to see available patterns, or provide a more specific description.",
            "available_patterns": list(self._common_patterns.keys())
        }

    def test(self, pattern: str, test_string: str) -> dict:
        """Test a regex pattern against a string.

        Args:
            pattern: The regex pattern to test
            test_string: The string to test against

        Returns:
            dict with:
                - success: bool
                - matches: List of all matches found
                - groups: Captured groups from first match
                - positions: Start/end positions of matches
                - match_count: Number of matches found

        Example:
            >>> tool = RegexBuilderTool()
            >>> result = tool.test(r'\\d+', 'abc 123 def 456')
            >>> result['matches']
            ['123', '456']
        """
        if not pattern:
            return {"success": False, "error": "No pattern provided"}
        if test_string is None:
            return {"success": False, "error": "No test string provided"}

        # Validate pattern first
        validation = self.validate(pattern)
        if not validation.get("valid"):
            return {
                "success": False,
                "error": f"Invalid pattern: {validation.get('error')}"
            }

        try:
            compiled = re.compile(pattern)
            matches = []
            positions = []
            groups_list = []

            for match in compiled.finditer(test_string):
                matches.append(match.group())
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "span": match.span()
                })
                if match.groups():
                    groups_list.append({
                        "groups": match.groups(),
                        "groupdict": match.groupdict() if match.groupdict() else None
                    })

            return {
                "success": True,
                "matches": matches,
                "match_count": len(matches),
                "positions": positions,
                "groups": groups_list if groups_list else None,
                "full_match": bool(compiled.fullmatch(test_string)),
                "pattern": pattern,
                "test_string": test_string
            }

        except re.error as e:
            return {
                "success": False,
                "error": f"Regex error: {str(e)}"
            }

    def explain(self, pattern: str) -> dict:
        """Explain a regex pattern in human-readable terms.

        Breaks down each component of the pattern and explains what it matches.

        Args:
            pattern: The regex pattern to explain

        Returns:
            dict with:
                - success: bool
                - pattern: The original pattern
                - explanation: Overall description
                - components: List of component explanations

        Example:
            >>> tool = RegexBuilderTool()
            >>> result = tool.explain(r'^[a-z]+$')
            >>> result['explanation']
            'Matches one or more lowercase letters from start to end of string'
        """
        if not pattern:
            return {"success": False, "error": "No pattern provided"}

        validation = self.validate(pattern)
        if not validation.get("valid"):
            return {
                "success": False,
                "error": f"Invalid pattern: {validation.get('error')}"
            }

        explanation = self._explain_pattern(pattern)
        components = self._break_down_components(pattern)

        return {
            "success": True,
            "pattern": pattern,
            "explanation": explanation,
            "components": components
        }

    def find_all(self, pattern: str, text: str) -> dict:
        """Find all matches of a pattern in text.

        Args:
            pattern: The regex pattern to search for
            text: The text to search in

        Returns:
            dict with:
                - success: bool
                - matches: List of matches with positions
                - count: Total number of matches
                - highlighted: Text with matches marked

        Example:
            >>> tool = RegexBuilderTool()
            >>> result = tool.find_all(r'\\d+', 'a1b2c3')
            >>> result['matches']
            [{'match': '1', 'start': 1, 'end': 2}, ...]
        """
        if not pattern:
            return {"success": False, "error": "No pattern provided"}
        if not text:
            return {"success": False, "error": "No text provided"}

        validation = self.validate(pattern)
        if not validation.get("valid"):
            return {
                "success": False,
                "error": f"Invalid pattern: {validation.get('error')}"
            }

        try:
            compiled = re.compile(pattern)
            matches = []

            for match in compiled.finditer(text):
                match_info = {
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end()
                }
                if match.groups():
                    match_info["groups"] = match.groups()
                if match.groupdict():
                    match_info["named_groups"] = match.groupdict()
                matches.append(match_info)

            # Create highlighted version (wrap matches in brackets)
            highlighted = compiled.sub(r'[\g<0>]', text)

            return {
                "success": True,
                "matches": matches,
                "count": len(matches),
                "highlighted": highlighted,
                "pattern": pattern
            }

        except re.error as e:
            return {
                "success": False,
                "error": f"Regex error: {str(e)}"
            }

    def replace(self, pattern: str, text: str, replacement: str) -> dict:
        """Replace all matches of a pattern in text.

        Args:
            pattern: The regex pattern to match
            text: The text to perform replacements in
            replacement: The replacement string (can use \\1, \\2 for groups)

        Returns:
            dict with:
                - success: bool
                - result: The modified text
                - count: Number of replacements made
                - original: The original text

        Example:
            >>> tool = RegexBuilderTool()
            >>> result = tool.replace(r'\\d+', 'a1b2c3', 'X')
            >>> result['result']
            'aXbXcX'
        """
        if not pattern:
            return {"success": False, "error": "No pattern provided"}
        if text is None:
            return {"success": False, "error": "No text provided"}
        if replacement is None:
            return {"success": False, "error": "No replacement provided"}

        validation = self.validate(pattern)
        if not validation.get("valid"):
            return {
                "success": False,
                "error": f"Invalid pattern: {validation.get('error')}"
            }

        try:
            compiled = re.compile(pattern)
            result, count = compiled.subn(replacement, text)

            return {
                "success": True,
                "result": result,
                "count": count,
                "original": text,
                "pattern": pattern,
                "replacement": replacement
            }

        except re.error as e:
            return {
                "success": False,
                "error": f"Regex error: {str(e)}"
            }

    def validate(self, pattern: str) -> dict:
        """Validate if a regex pattern is syntactically correct.

        Args:
            pattern: The regex pattern to validate

        Returns:
            dict with:
                - valid: bool
                - error: Error message if invalid
                - pattern: The validated pattern

        Example:
            >>> tool = RegexBuilderTool()
            >>> tool.validate(r'[a-z]+')
            {'valid': True, 'pattern': '[a-z]+'}
            >>> tool.validate(r'[a-z')
            {'valid': False, 'error': '...', 'pattern': '[a-z'}
        """
        if not pattern:
            return {
                "valid": False,
                "error": "Empty pattern",
                "pattern": pattern
            }

        try:
            re.compile(pattern)
            return {
                "valid": True,
                "pattern": pattern
            }
        except re.error as e:
            return {
                "valid": False,
                "error": str(e),
                "pattern": pattern,
                "position": e.pos if hasattr(e, 'pos') else None
            }

    def common_patterns(self) -> dict:
        """Return a dictionary of commonly used regex patterns.

        Returns:
            dict with:
                - success: bool
                - patterns: Dict of pattern_name -> pattern string
                - descriptions: Dict of pattern_name -> description

        Example:
            >>> tool = RegexBuilderTool()
            >>> result = tool.common_patterns()
            >>> result['patterns']['email']
            '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'
        """
        descriptions = {
            "email": "Match email addresses (user@domain.com)",
            "url": "Match URLs (http:// or https://)",
            "phone": "Match US phone numbers (various formats)",
            "phone_international": "Match international phone numbers",
            "ip_address": "Match IPv4 addresses (192.168.1.1)",
            "ipv6": "Match IPv6 addresses",
            "date_iso": "Match ISO dates (YYYY-MM-DD)",
            "date_us": "Match US dates (MM/DD/YYYY)",
            "date_eu": "Match European dates (DD/MM/YYYY)",
            "time_24h": "Match 24-hour time (14:30, 08:00:00)",
            "time_12h": "Match 12-hour time with AM/PM",
            "hex_color": "Match hex color codes (#FFF, #FFFFFF)",
            "credit_card": "Match credit card numbers (Visa, MC, Amex, Discover)",
            "ssn": "Match US Social Security Numbers (XXX-XX-XXXX)",
            "zip_code_us": "Match US ZIP codes (12345 or 12345-6789)",
            "username": "Match usernames (3-16 chars, alphanumeric, underscore, hyphen)",
            "password_strong": "Match strong passwords (8+ chars, upper, lower, digit, special)",
            "slug": "Match URL slugs (lowercase-with-hyphens)",
            "uuid": "Match UUIDs/GUIDs",
            "mac_address": "Match MAC addresses (XX:XX:XX:XX:XX:XX)",
            "html_tag": "Match HTML tags with content",
            "whitespace": "Match whitespace characters",
            "word": "Match word boundaries",
            "integer": "Match integers (positive and negative)",
            "float": "Match floating point numbers",
            "alphanumeric": "Match alphanumeric strings only",
        }

        return {
            "success": True,
            "patterns": self._common_patterns.copy(),
            "descriptions": descriptions,
            "count": len(self._common_patterns)
        }

    def _explain_pattern(self, pattern: str) -> str:
        """Generate a human-readable explanation of a regex pattern."""
        explanations = []

        # Common pattern explanations
        token_explanations = {
            r'^': 'Start of string',
            r'$': 'End of string',
            r'.': 'Any character except newline',
            r'*': 'Zero or more of previous',
            r'+': 'One or more of previous',
            r'?': 'Zero or one of previous (optional)',
            r'\d': 'Any digit (0-9)',
            r'\D': 'Any non-digit',
            r'\w': 'Any word character (letter, digit, underscore)',
            r'\W': 'Any non-word character',
            r'\s': 'Any whitespace (space, tab, newline)',
            r'\S': 'Any non-whitespace',
            r'\b': 'Word boundary',
            r'\B': 'Non-word boundary',
            r'[a-z]': 'Any lowercase letter',
            r'[A-Z]': 'Any uppercase letter',
            r'[0-9]': 'Any digit',
            r'[a-zA-Z]': 'Any letter',
            r'[a-zA-Z0-9]': 'Any alphanumeric character',
        }

        # Check for common complete patterns
        if pattern == self._common_patterns.get("email"):
            return "Matches email addresses: username@domain.extension"
        if pattern == self._common_patterns.get("url"):
            return "Matches URLs starting with http:// or https://"
        if pattern == self._common_patterns.get("phone"):
            return "Matches US phone numbers in various formats"
        if pattern == self._common_patterns.get("ip_address"):
            return "Matches IPv4 addresses (0-255.0-255.0-255.0-255)"

        # Build explanation from components
        if pattern.startswith('^'):
            explanations.append("Anchored at start of string")
        if pattern.endswith('$'):
            explanations.append("Anchored at end of string")

        # Check for character classes
        if '[' in pattern:
            classes = re.findall(r'\[([^\]]+)\]', pattern)
            for cls in classes:
                if cls == 'a-z':
                    explanations.append("lowercase letters")
                elif cls == 'A-Z':
                    explanations.append("uppercase letters")
                elif cls == '0-9':
                    explanations.append("digits")
                elif cls == 'a-zA-Z':
                    explanations.append("any letter")
                elif cls == 'a-zA-Z0-9':
                    explanations.append("alphanumeric characters")
                else:
                    explanations.append(f"characters: {cls}")

        # Check for quantifiers
        if '+' in pattern:
            explanations.append("one or more occurrences")
        if '*' in pattern:
            explanations.append("zero or more occurrences")
        if '?' in pattern and '(?:' not in pattern:
            explanations.append("optional element")

        if explanations:
            return "Matches " + ", ".join(explanations)
        return f"Regex pattern: {pattern}"

    def _break_down_components(self, pattern: str) -> list:
        """Break down a pattern into its components with explanations."""
        components = []

        # Token patterns to identify
        tokens = [
            (r'\^', 'Start anchor (^)', 'Matches start of string'),
            (r'\$', 'End anchor ($)', 'Matches end of string'),
            (r'\\d', 'Digit (\\d)', 'Matches any digit 0-9'),
            (r'\\D', 'Non-digit (\\D)', 'Matches any non-digit'),
            (r'\\w', 'Word char (\\w)', 'Matches letter, digit, or underscore'),
            (r'\\W', 'Non-word (\\W)', 'Matches non-word character'),
            (r'\\s', 'Whitespace (\\s)', 'Matches space, tab, newline'),
            (r'\\S', 'Non-whitespace (\\S)', 'Matches non-whitespace'),
            (r'\\b', 'Word boundary (\\b)', 'Matches position at word boundary'),
            (r'\.', 'Literal dot (\\.)', 'Matches a literal period'),
            (r'\.', 'Any char (.)', 'Matches any character except newline'),
            (r'\+', 'One or more (+)', 'Matches one or more of previous'),
            (r'\*', 'Zero or more (*)', 'Matches zero or more of previous'),
            (r'\?', 'Optional (?)', 'Matches zero or one of previous'),
            (r'\[([^\]]+)\]', 'Character class', 'Matches any character in brackets'),
            (r'\(([^)]+)\)', 'Capture group', 'Captures matched text'),
            (r'\{(\d+)\}', 'Exact count', 'Matches exactly N times'),
            (r'\{(\d+),(\d+)\}', 'Range count', 'Matches N to M times'),
            (r'\|', 'Alternation (|)', 'Matches either side of the bar'),
        ]

        # Simple component extraction
        i = 0
        while i < len(pattern):
            matched = False
            for token_pattern, name, desc in tokens:
                match = re.match(token_pattern, pattern[i:])
                if match:
                    components.append({
                        "token": match.group(),
                        "name": name,
                        "description": desc,
                        "position": i
                    })
                    i += len(match.group())
                    matched = True
                    break

            if not matched:
                # Single character
                components.append({
                    "token": pattern[i],
                    "name": f"Literal '{pattern[i]}'",
                    "description": f"Matches the character '{pattern[i]}'",
                    "position": i
                })
                i += 1

        return components

    def _generate_examples(self, pattern_name: str) -> list:
        """Generate example matches for a pattern type."""
        examples = {
            "email": ["user@example.com", "john.doe@company.co.uk", "test123@gmail.com"],
            "url": ["https://www.example.com", "http://api.test.io/path", "https://github.com/user/repo"],
            "phone": ["555-123-4567", "(555) 123-4567", "5551234567", "+1-555-123-4567"],
            "ip_address": ["192.168.1.1", "10.0.0.255", "172.16.0.1"],
            "date_iso": ["2024-01-15", "2023-12-31", "2025-06-30"],
            "date_us": ["01/15/2024", "12/31/2023", "06/30/2025"],
            "date_eu": ["15/01/2024", "31/12/2023", "30/06/2025"],
            "time_24h": ["14:30", "08:00:00", "23:59"],
            "time_12h": ["2:30 PM", "8:00 AM", "11:59 pm"],
            "hex_color": ["#FFF", "#FF5733", "#00FF00"],
            "uuid": ["550e8400-e29b-41d4-a716-446655440000"],
            "zip_code_us": ["12345", "12345-6789", "90210"],
            "username": ["john_doe", "user123", "test-user"],
            "slug": ["my-blog-post", "hello-world", "test-slug-123"],
        }
        return examples.get(pattern_name, [])

    def _build_basic_pattern(self, description: str) -> Optional[dict]:
        """Build a basic pattern from description keywords."""
        desc_lower = description.lower()

        # Simple keyword-based pattern generation
        if "start" in desc_lower and "end" in desc_lower:
            if "letter" in desc_lower:
                return {
                    "pattern": r"^[a-zA-Z]+$",
                    "explanation": "Matches strings containing only letters from start to end",
                    "examples": ["Hello", "world", "ABC"]
                }
            if "digit" in desc_lower or "number" in desc_lower:
                return {
                    "pattern": r"^\d+$",
                    "explanation": "Matches strings containing only digits from start to end",
                    "examples": ["123", "456789", "0"]
                }

        if "starts with" in desc_lower:
            # Extract what it should start with
            match = re.search(r"starts? with ['\"]?(\w+)['\"]?", desc_lower)
            if match:
                prefix = match.group(1)
                return {
                    "pattern": f"^{re.escape(prefix)}",
                    "explanation": f"Matches strings starting with '{prefix}'",
                    "examples": [f"{prefix}test", f"{prefix}123"]
                }

        if "ends with" in desc_lower:
            match = re.search(r"ends? with ['\"]?(\w+)['\"]?", desc_lower)
            if match:
                suffix = match.group(1)
                return {
                    "pattern": f"{re.escape(suffix)}$",
                    "explanation": f"Matches strings ending with '{suffix}'",
                    "examples": [f"test{suffix}", f"123{suffix}"]
                }

        if "contains" in desc_lower:
            match = re.search(r"contains? ['\"]?(\w+)['\"]?", desc_lower)
            if match:
                substring = match.group(1)
                return {
                    "pattern": re.escape(substring),
                    "explanation": f"Matches strings containing '{substring}'",
                    "examples": [f"pre{substring}post", substring]
                }

        if "only letter" in desc_lower or "letters only" in desc_lower:
            return {
                "pattern": r"^[a-zA-Z]+$",
                "explanation": "Matches strings containing only letters",
                "examples": ["Hello", "world"]
            }

        if "only digit" in desc_lower or "digits only" in desc_lower or "only number" in desc_lower:
            return {
                "pattern": r"^\d+$",
                "explanation": "Matches strings containing only digits",
                "examples": ["123", "456"]
            }

        return None

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a regex builder action.

        Args:
            action: Action to perform (build, test, explain, find_all, replace, validate, common_patterns)
            **kwargs: Additional arguments

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        # Common patterns request
        if "common" in action_lower or "list" in action_lower or "available" in action_lower:
            return self.common_patterns()

        # Validate pattern
        if "validate" in action_lower or "check" in action_lower or "valid" in action_lower:
            pattern = kwargs.get("pattern") or self._extract_pattern(action)
            if pattern:
                return self.validate(pattern)
            return {"success": False, "error": "No pattern to validate"}

        # Explain pattern
        if "explain" in action_lower or "what does" in action_lower or "meaning" in action_lower:
            pattern = kwargs.get("pattern") or self._extract_pattern(action)
            if pattern:
                return self.explain(pattern)
            return {"success": False, "error": "No pattern to explain"}

        # Test pattern
        if "test" in action_lower or "try" in action_lower:
            pattern = kwargs.get("pattern") or self._extract_pattern(action)
            test_string = kwargs.get("test_string") or kwargs.get("text") or self._extract_test_string(action)
            if pattern and test_string:
                return self.test(pattern, test_string)
            return {"success": False, "error": "Need both pattern and test string"}

        # Find all matches
        if "find" in action_lower or "search" in action_lower or "match" in action_lower:
            pattern = kwargs.get("pattern") or self._extract_pattern(action)
            text = kwargs.get("text") or self._extract_test_string(action)
            if pattern and text:
                return self.find_all(pattern, text)
            return {"success": False, "error": "Need both pattern and text"}

        # Replace
        if "replace" in action_lower or "substitute" in action_lower:
            pattern = kwargs.get("pattern") or self._extract_pattern(action)
            text = kwargs.get("text") or self._extract_test_string(action)
            replacement = kwargs.get("replacement", "")
            if pattern and text:
                return self.replace(pattern, text, replacement)
            return {"success": False, "error": "Need pattern, text, and replacement"}

        # Build pattern from description (default)
        return self.build(action)

    def _extract_pattern(self, action: str) -> Optional[str]:
        """Extract regex pattern from action string."""
        # Look for pattern in quotes
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            # Return first one that looks like a regex
            for q in quoted:
                if any(c in q for c in r'\[]().*+?^$|{}'):
                    return q
            return quoted[0]

        # Look for pattern after keywords
        patterns = [
            r'pattern[:\s]+(\S+)',
            r'regex[:\s]+(\S+)',
        ]
        for p in patterns:
            match = re.search(p, action, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_test_string(self, action: str) -> Optional[str]:
        """Extract test string from action."""
        # Look for text after "against", "in", "on"
        patterns = [
            r'against ["\']([^"\']+)["\']',
            r'in ["\']([^"\']+)["\']',
            r'on ["\']([^"\']+)["\']',
            r'text[:\s]+["\']([^"\']+)["\']',
            r'string[:\s]+["\']([^"\']+)["\']',
        ]
        for p in patterns:
            match = re.search(p, action, re.IGNORECASE)
            if match:
                return match.group(1)
        return None


# Singleton instance for easy import
regex_builder_tool = RegexBuilderTool()
