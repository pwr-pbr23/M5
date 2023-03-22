package org.apache.lucene.analysis.el;

/**
* Copyright 2005 The Apache Software Foundation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;  // for javadoc
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;

/**
* {@link Analyzer} for the Greek language.
* <p>
* Supports an external list of stopwords (words
* that will not be indexed at all).
* A default set of stopwords is used unless an alternative list is specified.
* </p>
*
* <p><b>NOTE</b>: This class uses the same {@link Version}
* dependent settings as {@link StandardAnalyzer}.</p>
*/
public final class GreekAnalyzer extends Analyzer
{
/**
* List of typical Greek stopwords.
*/
private static final String[] GREEK_STOP_WORDS = {
"Î¿", "Î·", "ÏÎ¿", "Î¿Î¹", "ÏÎ±", "ÏÎ¿Ï", "ÏÎ·Ï", "ÏÏÎ½", "ÏÎ¿Î½", "ÏÎ·Î½", "ÎºÎ±Î¹",
"ÎºÎ¹", "Îº", "ÎµÎ¹Î¼Î±Î¹", "ÎµÎ¹ÏÎ±Î¹", "ÎµÎ¹Î½Î±Î¹", "ÎµÎ¹Î¼Î±ÏÏÎµ", "ÎµÎ¹ÏÏÎµ", "ÏÏÎ¿", "ÏÏÎ¿Î½",
"ÏÏÎ·", "ÏÏÎ·Î½", "Î¼Î±", "Î±Î»Î»Î±", "Î±ÏÎ¿", "Î³Î¹Î±", "ÏÏÎ¿Ï", "Î¼Îµ", "ÏÎµ", "ÏÏ",
"ÏÎ±ÏÎ±", "Î±Î½ÏÎ¹", "ÎºÎ±ÏÎ±", "Î¼ÎµÏÎ±", "Î¸Î±", "Î½Î±", "Î´Îµ", "Î´ÎµÎ½", "Î¼Î·", "Î¼Î·Î½",
"ÎµÏÎ¹", "ÎµÎ½Ï", "ÎµÎ±Î½", "Î±Î½", "ÏÎ¿ÏÎµ", "ÏÎ¿Ï", "ÏÏÏ", "ÏÎ¿Î¹Î¿Ï", "ÏÎ¿Î¹Î±", "ÏÎ¿Î¹Î¿",
"ÏÎ¿Î¹Î¿Î¹", "ÏÎ¿Î¹ÎµÏ", "ÏÎ¿Î¹ÏÎ½", "ÏÎ¿Î¹Î¿ÏÏ", "Î±ÏÏÎ¿Ï", "Î±ÏÏÎ·", "Î±ÏÏÎ¿", "Î±ÏÏÎ¿Î¹",
"Î±ÏÏÏÎ½", "Î±ÏÏÎ¿ÏÏ", "Î±ÏÏÎµÏ", "Î±ÏÏÎ±", "ÎµÎºÎµÎ¹Î½Î¿Ï", "ÎµÎºÎµÎ¹Î½Î·", "ÎµÎºÎµÎ¹Î½Î¿",
"ÎµÎºÎµÎ¹Î½Î¿Î¹", "ÎµÎºÎµÎ¹Î½ÎµÏ", "ÎµÎºÎµÎ¹Î½Î±", "ÎµÎºÎµÎ¹Î½ÏÎ½", "ÎµÎºÎµÎ¹Î½Î¿ÏÏ", "Î¿ÏÏÏ", "Î¿Î¼ÏÏ",
"Î¹ÏÏÏ", "Î¿ÏÎ¿", "Î¿ÏÎ¹"
};

/**
* Returns a set of default Greek-stopwords
* @return a set of default Greek-stopwords
*/
public static final Set<?> getDefaultStopSet(){
return DefaultSetHolder.DEFAULT_SET;
}

private static class DefaultSetHolder {
private static final Set<?> DEFAULT_SET = CharArraySet.unmodifiableSet(new CharArraySet(
Arrays.asList(GREEK_STOP_WORDS), false));
}

/**
* Contains the stopwords used with the {@link StopFilter}.
*/
private final Set<?> stopSet;

private final Version matchVersion;

public GreekAnalyzer(Version matchVersion) {
this(matchVersion, DefaultSetHolder.DEFAULT_SET);
}

/**
* Builds an analyzer with the given stop words
*
* @param matchVersion
*          lucene compatibility version
* @param stopwords
*          a stopword set
*/
public GreekAnalyzer(Version matchVersion, Set<?> stopwords) {
stopSet = CharArraySet.unmodifiableSet(CharArraySet.copy(stopwords));
this.matchVersion = matchVersion;
}

/**
* Builds an analyzer with the given stop words.
* @param stopwords Array of stopwords to use.
* @deprecated use {@link #GreekAnalyzer(Version, Set)} instead
*/
public GreekAnalyzer(Version matchVersion, String... stopwords)
{
this(matchVersion, StopFilter.makeStopSet(stopwords));
}

/**
* Builds an analyzer with the given stop words.
* @deprecated use {@link #GreekAnalyzer(Version, Set)} instead
*/
public GreekAnalyzer(Version matchVersion, Map<?,?> stopwords)
{
this(matchVersion, stopwords.keySet());
}

/**
* Creates a {@link TokenStream} which tokenizes all the text in the provided {@link Reader}.
*
* @return  A {@link TokenStream} built from a {@link StandardTokenizer} filtered with
*                  {@link GreekLowerCaseFilter} and {@link StopFilter}
*/
@Override
public TokenStream tokenStream(String fieldName, Reader reader)
{
TokenStream result = new StandardTokenizer(matchVersion, reader);
result = new GreekLowerCaseFilter(result);
result = new StopFilter(StopFilter.getEnablePositionIncrementsVersionDefault(matchVersion),
result, stopSet);
return result;
}

private class SavedStreams {
Tokenizer source;
TokenStream result;
};

/**
* Returns a (possibly reused) {@link TokenStream} which tokenizes all the text
* in the provided {@link Reader}.
*
* @return  A {@link TokenStream} built from a {@link StandardTokenizer} filtered with
*                  {@link GreekLowerCaseFilter} and {@link StopFilter}
*/
@Override
public TokenStream reusableTokenStream(String fieldName, Reader reader)
throws IOException {
SavedStreams streams = (SavedStreams) getPreviousTokenStream();
if (streams == null) {
streams = new SavedStreams();
streams.source = new StandardTokenizer(matchVersion, reader);
streams.result = new GreekLowerCaseFilter(streams.source);
streams.result = new StopFilter(StopFilter.getEnablePositionIncrementsVersionDefault(matchVersion),
streams.result, stopSet);
setPreviousTokenStream(streams);
} else {
streams.source.reset(reader);
}
return streams.result;
}
}