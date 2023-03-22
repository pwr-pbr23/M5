package org.apache.lucene.analysis.ru;

/**
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.util.Version;

/**
* {@link Analyzer} for Russian language.
* <p>
* Supports an external list of stopwords (words that
* will not be indexed at all).
* A default set of stopwords is used unless an alternative list is specified.
* </p>
*/
public final class RussianAnalyzer extends Analyzer
{
/**
* List of typical Russian stopwords.
*/
private static final String[] RUSSIAN_STOP_WORDS = {
"Ð°", "Ð±ÐµÐ·", "Ð±Ð¾Ð»ÐµÐµ", "Ð±Ñ", "Ð±ÑÐ»", "Ð±ÑÐ»Ð°", "Ð±ÑÐ»Ð¸", "Ð±ÑÐ»Ð¾", "Ð±ÑÑÑ", "Ð²",
"Ð²Ð°Ð¼", "Ð²Ð°Ñ", "Ð²ÐµÑÑ", "Ð²Ð¾", "Ð²Ð¾Ñ", "Ð²ÑÐµ", "Ð²ÑÐµÐ³Ð¾", "Ð²ÑÐµÑ", "Ð²Ñ", "Ð³Ð´Ðµ",
"Ð´Ð°", "Ð´Ð°Ð¶Ðµ", "Ð´Ð»Ñ", "Ð´Ð¾", "ÐµÐ³Ð¾", "ÐµÐµ", "ÐµÐ¹", "ÐµÑ", "ÐµÑÐ»Ð¸", "ÐµÑÑÑ",
"ÐµÑÐµ", "Ð¶Ðµ", "Ð·Ð°", "Ð·Ð´ÐµÑÑ", "Ð¸", "Ð¸Ð·", "Ð¸Ð»Ð¸", "Ð¸Ð¼", "Ð¸Ñ", "Ðº", "ÐºÐ°Ðº",
"ÐºÐ¾", "ÐºÐ¾Ð³Ð´Ð°", "ÐºÑÐ¾", "Ð»Ð¸", "Ð»Ð¸Ð±Ð¾", "Ð¼Ð½Ðµ", "Ð¼Ð¾Ð¶ÐµÑ", "Ð¼Ñ", "Ð½Ð°", "Ð½Ð°Ð´Ð¾",
"Ð½Ð°Ñ", "Ð½Ðµ", "Ð½ÐµÐ³Ð¾", "Ð½ÐµÐµ", "Ð½ÐµÑ", "Ð½Ð¸", "Ð½Ð¸Ñ", "Ð½Ð¾", "Ð½Ñ", "Ð¾", "Ð¾Ð±",
"Ð¾Ð´Ð½Ð°ÐºÐ¾", "Ð¾Ð½", "Ð¾Ð½Ð°", "Ð¾Ð½Ð¸", "Ð¾Ð½Ð¾", "Ð¾Ñ", "Ð¾ÑÐµÐ½Ñ", "Ð¿Ð¾", "Ð¿Ð¾Ð´", "Ð¿ÑÐ¸",
"Ñ", "ÑÐ¾", "ÑÐ°Ðº", "ÑÐ°ÐºÐ¶Ðµ", "ÑÐ°ÐºÐ¾Ð¹", "ÑÐ°Ð¼", "ÑÐµ", "ÑÐµÐ¼", "ÑÐ¾", "ÑÐ¾Ð³Ð¾",
"ÑÐ¾Ð¶Ðµ", "ÑÐ¾Ð¹", "ÑÐ¾Ð»ÑÐºÐ¾", "ÑÐ¾Ð¼", "ÑÑ", "Ñ", "ÑÐ¶Ðµ", "ÑÐ¾ÑÑ", "ÑÐµÐ³Ð¾", "ÑÐµÐ¹",
"ÑÐµÐ¼", "ÑÑÐ¾", "ÑÑÐ¾Ð±Ñ", "ÑÑÐµ", "ÑÑÑ", "ÑÑÐ°", "ÑÑÐ¸", "ÑÑÐ¾", "Ñ"
};

private static class DefaultSetHolder {
static final Set<?> DEFAULT_STOP_SET = CharArraySet
.unmodifiableSet(new CharArraySet(Arrays.asList(RUSSIAN_STOP_WORDS),
false));
}

/**
* Contains the stopwords used with the StopFilter.
*/
private final Set<?> stopSet;

private final Version matchVersion;

public RussianAnalyzer(Version matchVersion) {
this(matchVersion, DefaultSetHolder.DEFAULT_STOP_SET);
}

/**
* Builds an analyzer with the given stop words.
* @deprecated use {@link #RussianAnalyzer(Version, Set)} instead
*/
public RussianAnalyzer(Version matchVersion, String... stopwords) {
this(matchVersion, StopFilter.makeStopSet(stopwords));
}

/**
* Builds an analyzer with the given stop words
*
* @param matchVersion
*          lucene compatibility version
* @param stopwords
*          a stopword set
*/
public RussianAnalyzer(Version matchVersion, Set<?> stopwords){
stopSet = CharArraySet.unmodifiableSet(CharArraySet.copy(stopwords));
this.matchVersion = matchVersion;
}

/**
* Builds an analyzer with the given stop words.
* TODO: create a Set version of this ctor
* @deprecated use {@link #RussianAnalyzer(Version, Set)} instead
*/
public RussianAnalyzer(Version matchVersion, Map<?,?> stopwords)
{
this(matchVersion, stopwords.keySet());
}

/**
* Creates a {@link TokenStream} which tokenizes all the text in the
* provided {@link Reader}.
*
* @return  A {@link TokenStream} built from a
*   {@link RussianLetterTokenizer} filtered with
*   {@link RussianLowerCaseFilter}, {@link StopFilter},
*   and {@link RussianStemFilter}
*/
@Override
public TokenStream tokenStream(String fieldName, Reader reader)
{
TokenStream result = new RussianLetterTokenizer(reader);
result = new LowerCaseFilter(result);
result = new StopFilter(StopFilter.getEnablePositionIncrementsVersionDefault(matchVersion),
result, stopSet);
result = new RussianStemFilter(result);
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
* @return  A {@link TokenStream} built from a
*   {@link RussianLetterTokenizer} filtered with
*   {@link RussianLowerCaseFilter}, {@link StopFilter},
*   and {@link RussianStemFilter}
*/
@Override
public TokenStream reusableTokenStream(String fieldName, Reader reader)
throws IOException {
SavedStreams streams = (SavedStreams) getPreviousTokenStream();
if (streams == null) {
streams = new SavedStreams();
streams.source = new RussianLetterTokenizer(reader);
streams.result = new LowerCaseFilter(streams.source);
streams.result = new StopFilter(StopFilter.getEnablePositionIncrementsVersionDefault(matchVersion),
streams.result, stopSet);
streams.result = new RussianStemFilter(streams.result);
setPreviousTokenStream(streams);
} else {
streams.source.reset(reader);
}
return streams.result;
}
}