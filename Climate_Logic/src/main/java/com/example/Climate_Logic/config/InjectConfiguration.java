package com.example.Climate_Logic.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.jackson.Jackson2ObjectMapperBuilderCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.TimeZone;

@Slf4j
@Configuration
public class InjectConfiguration {

    /**
     * Bean necesario para que jackson serialize y desserialize los tipos de datos Date, usando el timezone de la jvm
     *
     * @return
     */
    @Bean
    public Jackson2ObjectMapperBuilderCustomizer jacksonObjectMapperCustomization() {
        return jacksonObjectMapperBuilder ->
                jacksonObjectMapperBuilder.timeZone(TimeZone.getDefault());
    }
}

